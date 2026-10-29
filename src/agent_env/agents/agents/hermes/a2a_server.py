"""A2A agent wrapping Hermes Agent.

Built on agentenv-protocol's `AgentEnvAgent`: the framework serves the Agent Card, the A2A task lifecycle and the
SDK-owned extensions (agent-config, mcp-config, skill listing, task trajectories, peer agents). Each turn is one
`hermes agentenv turn` in the context's own Hermes profile, whose hook events are appended to the framework's
trajectory log as Hermes writes them. A context's first turn starts a Hermes session, which later turns resume.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from weakref import WeakValueDictionary

import httpx
from a2a.types import AgentSkill
from agentenv_protocol.a2a_agent import (
    ATTRIBUTION_PROBE_V1,
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

from hermes_bridge import (
    ERROR_MESSAGE_LIMIT,
    HERMES_ROOT,
    get_last_attribution_snapshot,
    kill_running_clis,
    profile_dir,
    run_hermes,
    validate_output_format,
)
from hermes_kit import DurableSessionRefs, in_thread_to_its_end, install_skill, json_error_bodies, to_its_end

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skills agent-env installs, which every context's profile reads as an external skills folder.
SKILLS_DIR = Path(os.environ.get("HERMES_AGENTENV_SKILLS_DIR") or HERMES_ROOT / "agentenv-skills")
WORKSPACE_DIR = Path(os.environ.get("HERMES_WORKSPACE_DIR", "/app"))
INPUTS_DIR = WORKSPACE_DIR / "inputs"
_FILE_DOWNLOAD_TIMEOUT_S = 60
_PDF_RENDER_TIMEOUT_S = 60
_PDF_RENDER_DPI = 150
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
# The versioned format of the Hermes hook events this agent appends to its trajectory.
TRAJECTORY_FORMAT = "hermes-hook-events/1"
# A snapshot's conversation object: the context's profile, all but what Hermes rebuilds or this agent rewrites.
_SNAPSHOT_EXCLUDED = frozenset({"cache", "logs", "plugins", "skills", "config.yaml", "audio_cache", "image_cache"})
_SNAPSHOT_SESSION = "agentenv-session.json"


class HermesConfig(AgentConfig):
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
    """A supplied file cannot be passed to Hermes."""


def _safe_name(candidate: str | None, fallback_ext: str = "") -> str:
    if candidate:
        candidate = _SAFE_NAME_RE.sub("_", Path(candidate).name)
    return candidate or f"{uuid.uuid4().hex}{fallback_ext}"


async def _download_uri_to(uri: str, target: Path) -> None:
    """Write an http(s) file to target as it arrives, so a large one never sits whole in memory; a cancelled turn's
    download stops with it."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        raise FilePartMaterializeError(f"unsupported file URI scheme {parsed.scheme!r}")
    try:
        async with httpx.AsyncClient(timeout=_FILE_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
            async with client.stream("GET", uri) as response:
                response.raise_for_status()
                with target.open("wb") as out:
                    async for chunk in response.aiter_bytes():
                        out.write(chunk)
    except httpx.HTTPError as exc:
        raise FilePartMaterializeError("HTTP file download failed") from exc


def _write_base64(target: Path, data: str) -> None:
    target.write_bytes(base64.b64decode(data, validate=True))


async def _render_pdf(target: Path, prefix: str) -> bool:
    """Whether pdftoppm rendered the PDF's pages beside it; one that outlives its timeout, or a cancelled turn, is
    killed and reaped before this returns."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pdftoppm", "-png", "-r", str(_PDF_RENDER_DPI), str(target), prefix,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
    try:
        return await asyncio.wait_for(proc.wait(), _PDF_RENDER_TIMEOUT_S) == 0
    except TimeoutError:
        return False
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await to_its_end(proc.wait())


async def _materialize_and_render(part: FilePart, task_id: str) -> list[str]:
    """A file part's path under the task's inputs, and a PDF's page images; each write ends before this returns, even
    for a cancelled turn."""
    directory = INPUTS_DIR / task_id
    directory.mkdir(parents=True, exist_ok=True)
    name = _safe_name(part.name)
    target = directory / name
    if target.exists():
        target = directory / f"{uuid.uuid4().hex[:8]}-{name}"
    if part.bytes is not None:
        try:
            await in_thread_to_its_end(_write_base64, target, part.bytes)
        except (binascii.Error, ValueError, OSError) as exc:
            raise FilePartMaterializeError(f"invalid file attachment {name!r}") from exc
    else:
        await _download_uri_to(part.uri, target)
    if (part.mime_type or "").lower() != "application/pdf" and target.suffix.lower() != ".pdf":
        return [str(target)]
    # Hermes looks at images with its vision tool, so a PDF goes as its pages' images too: in a folder of their own, so
    # two PDFs whose names share a stem (report.pdf, report.PDF) never mix their pages.
    pages_dir = Path(tempfile.mkdtemp(prefix=f"{target.name}-pages-", dir=directory))
    if not await _render_pdf(target, str(pages_dir / "page")):
        logger.warning("PDF rendering failed for %s; passing the PDF alone", target)
        return [str(target)]
    pages = sorted(pages_dir.glob("page*.png"))
    return [str(target), *(str(page) for page in pages)]


def _with_attachments(prompt: str, paths: list[str]) -> str:
    """The prompt, naming the files the task attached, which Hermes reads with its file and vision tools."""
    if not paths:
        return prompt
    listing = "\n".join(f"- {path}" for path in paths)
    return f"{prompt}\n\nAttached files (read them, or look at images with your vision tool):\n{listing}"


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
    """As the server stops, the processes of turns still running are killed, which would otherwise go on in the
    workspace after it."""
    try:
        yield
    finally:
        if stopped := kill_running_clis():
            logger.info(f"Killed {stopped} hermes turn(s) still running as the agent shut down")


# A Hermes turn's process peaked near 210 MB running shell and Python tools, and its tools' own programs can take as
# much again; the server and the OS need about 1 GB more. Turns past the cap wait rather than run the machine out of
# memory.
TURN_MEMORY_MB, RESERVED_MEMORY_MB = 400, 1024


def max_concurrent_turns(environ: Mapping[str, str] = os.environ) -> int:
    """How many Hermes turns may run at once: HERMES_MAX_CONCURRENT_TURNS, else as many as memory fits."""
    if configured := environ.get("HERMES_MAX_CONCURRENT_TURNS"):
        return max(1, int(configured))
    return max(1, (memory_mb() - RESERVED_MEMORY_MB) // TURN_MEMORY_MB)


def memory_mb(cgroup_limit: Path = Path("/sys/fs/cgroup/memory.max")) -> int:
    """The memory this server's processes may use: the machine's, or its cgroup's limit when that's lower."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    try:
        limit = cgroup_limit.read_text().strip()
    except OSError:
        limit = "max"
    if limit.isdigit():
        total = min(total, int(limit))
    return total // 2**20


def _archive_profile(home: Path, session_id: str, archive: Path) -> None:
    """The context's conversation as a tar.gz: its profile, all but what Hermes rebuilds, with its session id."""
    (home / _SNAPSHOT_SESSION).write_text(json.dumps({"session_id": session_id}))
    try:
        with tarfile.open(archive, "w:gz") as tar:
            for child in sorted(home.iterdir()):
                if child.name not in _SNAPSHOT_EXCLUDED:
                    tar.add(child, arcname=child.name)
    finally:
        (home / _SNAPSHOT_SESSION).unlink(missing_ok=True)


def _stage_profile(archive: Path, home: Path) -> tuple[Path, str]:
    """An archived profile extracted beside the context's, which stays untouched; the staged folder and the archived
    session id."""
    staged = Path(tempfile.mkdtemp(prefix=".restore-", dir=home.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staged, filter="data")
        session_id = json.loads((staged / _SNAPSHOT_SESSION).read_text()).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("the snapshot names no Hermes session")
        (staged / _SNAPSHOT_SESSION).unlink()
        return staged, session_id
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def _commit_profile(staged: Path, home: Path) -> None:
    """Swap a staged profile in for the context's; the previous profile comes back if the move fails."""
    previous = home.with_name(home.name + ".previous")
    shutil.rmtree(previous, ignore_errors=True)
    if home.exists():
        home.rename(previous)
    try:
        staged.rename(home)
    except BaseException:
        shutil.rmtree(home, ignore_errors=True)
        if previous.exists():
            previous.rename(home)
        raise
    # The swap has succeeded; removing the old profile is cleanup and must not undo it.
    shutil.rmtree(previous, ignore_errors=True)


def _tar(*args: str) -> None:
    subprocess.run(["tar", *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


@a2a_agent(
    identity=AgentIdentity(
        name="Hermes Agent",
        description="A2A-compliant wrapper around Nous Research's Hermes Agent.",
        version="1.0.0",
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"),
        output_modes=("text",),
        skills=(AgentSkill(
            id="hermes", name="hermes",
            description="Runs Hermes Agent with its full toolset (terminal, files, code execution, web, skills, "
                        "memory, subagents).",
            tags=["coding", "agent"],
        ),),
    ),
    config=HermesConfig,
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True),
        enable(ATTRIBUTION_PROBE_V1),
        # SDK default: Hermes reaches the task's other agents through the SDK's peer_agents MCP tools.
        PEER_AGENTS_V1,
    ),
    lifespan=lifespan,
    workspace=str(WORKSPACE_DIR),
)
class HermesAgent(DurableSessionRefs, AgentEnvAgent):
    """Bridges A2A tasks onto `hermes agentenv turn`."""

    def __init__(self) -> None:
        self._turn_slots = asyncio.Semaphore(max_concurrent_turns())
        self._turn_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._workspace = _WorkspaceBarrier()

    def _turn_lock(self, context_id: str) -> asyncio.Lock:
        return self._turn_locks.setdefault(context_id, asyncio.Lock())

    async def run(self, request: TaskRequest[HermesConfig]) -> TaskResult:
        # The plugin's hook events, appended as Hermes writes them, are the turn's trajectory: readable while it runs,
        # and its final record once it ends.
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        prompt = "\n".join(part.text for part in request.parts if isinstance(part, TextPart) and part.text)
        context_id = request.context_id
        async with self._workspace.shared(), self._turn_lock(context_id):
            # Read under the lock rather than from the request: a snapshot load that held it may have just bound this
            # context to a restored session.
            session_id = self.session_ref_for_context(context_id)

            def record(event: dict) -> None:
                request.trajectory.append(event)
                # A first turn's session is bound as soon as Hermes names it, so a cancel can't lose it.
                started = event.get("session_id")
                if self.session_ref_for_context(context_id) is None and isinstance(started, str) and started:
                    self.set_session_ref_for_context(context_id, started)

            # Attachments count against the turn limit too: their downloads and PDF renders take memory and processes.
            async with self._turn_slots:
                try:
                    attachments = []
                    for part in request.parts:
                        if isinstance(part, FilePart):
                            attachments.extend(await _materialize_and_render(part, request.task_id))
                except FilePartMaterializeError as exc:
                    return TaskResult.failure("filepart_error", str(exc), error_type="infra_error")
                try:
                    result = await run_hermes(
                        prompt=_with_attachments(prompt, attachments),
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
                        skills_dir=SKILLS_DIR,
                        event_callback=record,
                        workspace=str(WORKSPACE_DIR),
                    )
                except Exception:
                    logger.exception("Hermes invocation failed")
                    return TaskResult.failure("hermes_invocation_error", "Hermes invocation failed",
                                              error_type="infra_error")
            if result.session_id:
                self.set_session_ref_for_context(context_id, result.session_id)

        parts = [TextPart(text=result.response or result.error or "")]
        usage = _usage(result)
        if result.error:
            details = {"error_type": "cli_error", "error_message": result.error[:ERROR_MESSAGE_LIMIT]}
            if result.error_code:
                details["error_code"] = result.error_code
            parts.append(DataPart(data=details))
            return (
                TaskResult.builder()
                .failed(result.error_code or "hermes_cli_error", result.error[:ERROR_MESSAGE_LIMIT])
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
            response.add_structured_output(result.structured_output)
        return response.build()

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(ATTRIBUTION_PROBE_V1.probe)
    async def probe_attribution(self) -> AttributionProbeResponse:
        """The LiteLLM attribution the last turn sent with its model calls."""
        snapshot = get_last_attribution_snapshot()
        return AttributionProbeResponse(
            last_seen_attribution={
                key: value for key, value in snapshot.items() if key != "last_seen_at_utc" and isinstance(value, str)
            },
            last_seen_at_utc=snapshot.get("last_seen_at_utc"),
        )

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="hermes-snapshot-") as directory:
            conversation = Path(directory) / "conversation.tgz"
            workspace = Path(directory) / "workspace.tgz"
            async with self._workspace.exclusive(), self._turn_lock(request.context_id):
                # Read once no turn runs: a context's first turn, or a load into it, may be about to bind it.
                session_id = self.session_ref_for_context(request.context_id)
                if not session_id:
                    raise HTTPException(status_code=404, detail="No active session for context_id")
                # Each to its end even for a cancelled save, so neither writes past the exclusive hold.
                await in_thread_to_its_end(_archive_profile, profile_dir(request.context_id), session_id, conversation)
                if request.objects.workspace is not None:
                    await in_thread_to_its_end(_tar, "-czf", str(workspace), "-C", str(WORKSPACE_DIR), ".")
            objects = {"trajectory": (await upload(request.objects.trajectory, conversation)).model_dump(mode="json")}
            if request.objects.workspace is not None:
                objects["workspace"] = (await upload(request.objects.workspace, workspace)).model_dump(mode="json")
        return {"context_id": request.context_id, "objects": objects}

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        target_context_id = request.target_context_id or uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="hermes-restore-") as directory:
            conversation = Path(directory) / "conversation.tgz"
            await download(request.objects.trajectory, conversation)
            staged = None
            if request.objects.workspace is not None:
                archive = Path(directory) / "workspace.tgz"
                staged = Path(directory) / "workspace"
                staged.mkdir()
                await download(request.objects.workspace, archive)
                await in_thread_to_its_end(_tar, "-xzf", str(archive), "-C", str(staged))
            async with self._workspace.exclusive(), self._turn_lock(target_context_id):
                home = profile_dir(target_context_id)
                home.parent.mkdir(parents=True, exist_ok=True)
                # Stage the conversation first, so a bad snapshot changes nothing. The workspace and conversation swaps
                # then run in one thread, and the context is bound to the restored session, in one step a cancel can't
                # abandon: a failed conversation swap puts the workspace back, and the context's previous profile stays
                # in place until both succeed.
                try:
                    staged_home, session_id = await in_thread_to_its_end(_stage_profile, conversation, home)
                except (tarfile.TarError, OSError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=f"Unreadable Hermes snapshot: {exc}") from exc
                commit = functools.partial(_commit_profile, staged_home, home)

                async def swap() -> None:
                    try:
                        if staged is not None:
                            await asyncio.to_thread(_replace_workspace, staged, commit)
                        else:
                            await asyncio.to_thread(commit)
                    finally:
                        await asyncio.to_thread(shutil.rmtree, staged_home, True)
                    self.set_session_ref_for_context(target_context_id, session_id)

                await to_its_end(swap())
        return {"context_id": target_context_id}


def _usage(result) -> Usage:
    """The turn's tool calls and the token counts and cost Hermes reported for its main model calls."""
    report = result.usage or {}

    def count(name: str) -> int | None:
        value = report.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    cost = report.get("estimated_cost_usd")
    details = {key: report[key] for key in ("api_calls", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
                                             "cost_status", "turn_exit_reason") if report.get(key) is not None}
    return Usage(
        tool_call_count=result.tool_call_count,
        input_tokens=count("input_tokens"),
        output_tokens=count("output_tokens"),
        total_tokens=count("total_tokens"),
        cost_usd=cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0 else None,
        provider_details=details,
    )


def _replace_workspace(staged: Path, then: Callable[[], None] | None = None) -> None:
    """Swap staged in for the workspace's contents. When then() raises after the swap, the previous contents come
    back."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".hermes-backup-", dir=WORKSPACE_DIR))
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
            if then is not None:
                then()
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


def build_app(agent: HermesAgent) -> Starlette:
    """The agent's Starlette app, answering errors with FastAPI's `{"detail": ...}` bodies."""
    return json_error_bodies(agent.create_app())


app = build_app(HermesAgent())
# agent-env reads the card from the image by importing this name.
AGENT_CARD = app.state.agentenv_a2a.card
