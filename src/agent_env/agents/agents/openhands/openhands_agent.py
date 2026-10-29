"""OpenHands (https://github.com/OpenHands/software-agent-sdk) as an AgentEnv A2A agent.

Each task is one run of an OpenHands SDK conversation in the workspace. A context maps to one persisted conversation,
so later tasks in the context resume it. This module is the A2A side and imports no SDK; ``openhands_runtime`` runs
the conversation.
"""

import asyncio
import base64
import io
import json
import mimetypes
import shutil
import tarfile
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, Self
from urllib.parse import urlsplit

import httpx
from agentenv_protocol import ROLE_HEADER
from agentenv_protocol.a2a_agent import (
    INSTALL_V1,
    MCP_CONFIG_V1,
    PEER_AGENTS_V1,
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
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    SKILL_CONFIG_V1,
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
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

OPENHANDS_VERSION: Final[str] = "1.58.2"
TRAJECTORY_FORMAT: Final[str] = "openhands-events/v1"
# The model when agent-config sets none, as for a peer that other agents message directly.
DEFAULT_MODEL_ENV: Final[str] = "OPENHANDS_A2A_MODEL"
A2A_PORT: Final[int] = 8000
INSTALL_DIR: Final[str] = "/opt/openhands-a2a"
ENV_FILE: Final[str] = "agent.env"
BASE_STATE: Final[str] = "base_state.json"
# Where a conversation's archive keeps the files attached to it, and the directory they were saved in.
INPUTS: Final[str] = ".openhands-a2a-inputs"
INPUTS_SOURCE: Final[str] = ".openhands-a2a-inputs.json"
FETCH_TIMEOUT_SECONDS: Final[int] = 120
MAX_ATTACHMENT_BYTES: Final[int] = 256 * 1024 * 1024
# A conversation id is derived from its context id, so a context finds its conversation across restarts.
_CONVERSATIONS: Final[uuid.UUID] = uuid.uuid5(uuid.NAMESPACE_URL, "urn:agentenv:agents:openhands")
# Images go to the model as images as well as to disk; any other file is left on disk for the agent's tools.
_IMAGE_TYPES: Final[frozenset[str]] = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_NO_BODY_STATUSES: Final[frozenset[int]] = frozenset({204, 205, 304})

# sudo only where it needs no password: VM sandboxes have it, a local sandbox runs docker as the user.
_DOCKER: Final[str] = "$(sudo -n true 2>/dev/null && echo sudo) docker"
INSTALL_COMMANDS: Final[list[str]] = [
    f"{_DOCKER} cp {{agent_ctx_tar}} {{container}}:/tmp/openhands-a2a-ctx.tar.gz",
    f"{_DOCKER} exec {{container}} sh -c 'mkdir -p {INSTALL_DIR} && tar -xf /tmp/openhands-a2a-ctx.tar.gz -C {INSTALL_DIR}"
    " && rm /tmp/openhands-a2a-ctx.tar.gz'",
    f"{_DOCKER} exec {{container}} sh {INSTALL_DIR}/install.sh",
    # The credentials travel on a heredoc into a 0600 file: the installer logs only a command's first line,
    # and neither a command line nor the container's config ever holds them.
    f"{_DOCKER} exec -i {{container}} sh -c 'umask 077 && cat > {INSTALL_DIR}/{ENV_FILE}' <<'OPENHANDS_A2A_ENV'\n"
    "LITELLM_API_KEY={litellm_api_key}\n"
    "LITELLM_BASE_URL={litellm_base_url}\n"
    "LITELLM_EXTRA_HEADERS={litellm_extra_headers}\n"
    "OPENHANDS_A2A_ENV",
    f"{_DOCKER} exec -d -e A2A_PORT={{a2a_port}} {{container}} sh {INSTALL_DIR}/start.sh",
]


class OpenHandsConfig(AgentConfig):
    model: str | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    max_iterations: int = 500
    timeout_seconds: int = 1800


@dataclass(frozen=True, slots=True)
class Endpoint:
    """The LiteLLM gateway the model is called through, with the headers every call carries."""

    base_url: str
    api_key: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> Self | None:
        """The endpoint the ``LITELLM_*`` variables name; None without a base URL, which fails each task instead."""
        base_url = environ.get("LITELLM_BASE_URL")
        if not base_url:
            return None
        try:
            headers = json.loads(environ.get("LITELLM_EXTRA_HEADERS") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"LITELLM_EXTRA_HEADERS is not valid JSON: {exc}") from exc
        if not isinstance(headers, dict):
            raise ValueError("LITELLM_EXTRA_HEADERS must be a JSON object of header names to values")
        return cls(base_url, environ.get("LITELLM_API_KEY"), headers)


@dataclass(frozen=True, slots=True)
class Turn:
    """One task, as the runtime runs it."""

    conversation_id: str
    conversations_dir: Path
    workspace: Path
    text: str
    images: tuple[str, ...]
    mcp_servers: Mapping[str, Any]
    skills_dir: Path
    skill_names: tuple[str, ...]
    config: OpenHandsConfig
    endpoint: Endpoint
    on_event: Callable[[dict[str, Any]], None]
    # What a run that was cancelled had spent by then.
    on_usage: Callable[[Usage], None]


@dataclass(frozen=True, slots=True)
class Outcome:
    text: str
    usage: Usage
    error_code: str | None = None
    error_type: Literal["agent_error", "infra_error"] = "agent_error"


@dataclass(slots=True)
class _ConversationLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    holders: int = 0


class Runtime(Protocol):
    async def run(self, turn: Turn) -> Outcome:
        """Run ``turn`` to its end. Cancelled, it interrupts the conversation and re-raises once it has stopped."""
        ...

    def stop(self) -> None:
        """Interrupt every turn still running: the server is shutting down."""
        ...


async def _detail_body(_request: Request, exc: HTTPException) -> Response:
    """An ``HTTPException`` as the ``{"detail": ...}`` body agent-env reads errors from."""
    if exc.status_code < 200 or exc.status_code in _NO_BODY_STATUSES:
        return Response(status_code=exc.status_code, headers=exc.headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@asynccontextmanager
async def _turns_stop_with_the_server(app: Starlette) -> AsyncIterator[None]:
    """The server's lifespan: as it shuts down, turns still running are interrupted, so their runs, in threads of
    their own, do not keep the process alive after the server."""
    try:
        yield
    finally:
        app.state.agentenv_a2a.agent.runtime.stop()


def conversation_id(context_id: str) -> str:
    return uuid.uuid5(_CONVERSATIONS, context_id).hex


def _with_role(mcp_servers: Mapping[str, Any], role: str | None) -> dict[str, Any]:
    """The servers with the role header, which the env gateway filters tools by and which wins over a stored role."""
    if not role:
        return dict(mcp_servers)
    return {
        name: {**registration, "headers": {**(registration.get("headers") or {}), ROLE_HEADER: role}}
        for name, registration in mcp_servers.items()
    }


def _media_type(part: FilePart) -> str:
    return part.mime_type or mimetypes.guess_type(part.name or "")[0] or "application/octet-stream"


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


def _pack(directory: Path, archive: Path, inputs: Path | None = None) -> None:
    """``directory``'s contents and, for a conversation, the files attached to it, from ``inputs``, under ``INPUTS``,
    with the directory they were saved in."""
    with tarfile.open(archive, "w:gz") as tar:
        for child in sorted(directory.iterdir()):
            tar.add(child, arcname=child.name)
        if inputs is not None and inputs.is_dir():
            tar.add(inputs, arcname=INPUTS)
            source = json.dumps({"directory": str(inputs)}).encode()
            info = tarfile.TarInfo(INPUTS_SOURCE)
            info.size = len(source)
            tar.addfile(info, io.BytesIO(source))


def _extract(archive: Path, directory: Path) -> Path:
    """The archive extracted into a new staging directory inside ``directory``."""
    staging = directory / f".openhands-a2a-restore-{uuid.uuid4().hex}"
    # An empty workspace's archive has no members, and extracting it creates nothing.
    staging.mkdir()
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def _replace_contents(directory: Path, staging: Path) -> None:
    """Replace ``directory``'s contents with those of ``staging``, a directory inside it."""
    for child in directory.iterdir():
        if child == staging:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in staging.iterdir():
        child.rename(directory / child.name)
    staging.rmdir()


def _repoint(directory: Path, old: str, new: str, skip: Path) -> None:
    """Make the files in ``directory``, but for those under ``skip``, name paths under ``new`` where they name paths
    under ``old``, whether written as is or JSON-escaped."""
    replacements = {f"{old}/": f"{new}/", f"{json.dumps(old)[1:-1]}/": f"{json.dumps(new)[1:-1]}/"}
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.is_relative_to(skip):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        repointed = text
        for before, after in replacements.items():
            repointed = repointed.replace(before, after)
        if repointed != text:
            path.write_text(repointed, encoding="utf-8")


def _stage_conversation(archive: Path, conversations_dir: Path, target_id: str, inputs: Path) -> Path:
    """The archived conversation extracted beside the others, rewritten to be ``target_id``'s with its attached files
    in ``inputs``: the SDK resumes a conversation only under the id and directory its state records, and its messages
    name the files by path. Raises ValueError for an archive that holds no conversation."""
    staging = _extract(archive, conversations_dir)
    try:
        state = json.loads((staging / BASE_STATE).read_text())
        uuid.UUID(state["id"])
        if not isinstance(state.get("agent"), dict):
            raise ValueError("its state names no agent")
        state.update(id=str(uuid.UUID(target_id)), persistence_dir=str(conversations_dir / target_id))
        source = staging / INPUTS_SOURCE
        saved_in = json.loads(source.read_text())["directory"] if source.is_file() else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"the trajectory object holds no OpenHands conversation: {type(exc).__name__}: {exc}") from exc
    (staging / BASE_STATE).write_text(json.dumps(state))
    if saved_in is not None:
        source.unlink()
        _repoint(staging, saved_in, str(inputs), skip=staging / INPUTS)
    return staging


@a2a_agent(
    identity=AgentIdentity(
        name="OpenHands",
        description="The OpenHands agent: reads, edits and runs code in its workspace with MCP tools and skills.",
        version=OPENHANDS_VERSION,
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "image/webp"),
    ),
    config=OpenHandsConfig,
    config_description="Configure the OpenHands runtime: model, effort, system prompt and iteration limit.",
    lifespan=_turns_stop_with_the_server,
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True, description="The task's OpenHands SDK events, also while it runs."),
        enable(
            SNAPSHOT_V1,
            description=(
                "The trajectory object is the context's persisted OpenHands conversation, the workspace object a "
                "gzipped tar."
            ),
        ),
        PEER_AGENTS_V1,
        enable(
            INSTALL_V1,
            install_commands=INSTALL_COMMANDS,
            required_params=[
                "container", "agent_ctx_tar", "a2a_port", "litellm_api_key", "litellm_base_url",
                "litellm_extra_headers",
            ],
            a2a_port=A2A_PORT,
        ),
        TRIGGERS_V1,
    ),
)
class OpenHandsAgent(AgentEnvAgent):
    def __init__(
        self,
        runtime: Runtime,
        *,
        endpoint: Endpoint | None,
        default_model: str | None = None,
        home: Path | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.runtime: Runtime = runtime
        self.endpoint: Endpoint | None = endpoint
        self.default_model: str | None = default_model
        self.home: Path = home or Path.home() / ".openhands-a2a"
        self.workspace: Path = workspace or Path("/workspace")
        self.conversations_dir: Path = self.home / "conversations"
        self.skills_dir: Path = self.home / "skills"
        # The files attached in each conversation, in a directory of its own, which its snapshots carry.
        self.inputs_dir: Path = self.home / "inputs"
        # The framework already runs a context's tasks one at a time; this lets a snapshot save wait for the task
        # running in its context. An entry lives only while it is held, so contexts never seen again leave nothing.
        self._conversation_locks: dict[str, _ConversationLock] = {}
        # A task starts once no restore or workspace save holds this. A restore is refused while tasks run, and a
        # workspace save waits for the tasks running in other contexts to end.
        self._workspace_lock: asyncio.Lock = asyncio.Lock()
        self._running: int = 0
        self._idle: asyncio.Event = asyncio.Event()
        self._idle.set()

    def create_app(self) -> Starlette:
        app = super().create_app()
        app.add_exception_handler(HTTPException, _detail_body)
        return app

    async def run(self, request: TaskRequest[OpenHandsConfig]) -> TaskResult:
        config = request.config
        if not config.model and self.default_model:
            config = config.model_copy(update={"model": self.default_model})
        if not config.model:
            return TaskResult.failure("openhands.no_model", "no model is configured; set 'model' through agent-config")
        if self.endpoint is None:
            return TaskResult.failure("openhands.no_endpoint", "LITELLM_BASE_URL is not set", error_type="infra_error")
        workspace = request.workspace or self.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        # The SDK calls back from its own threads too; the log takes one appender at a time.
        appending = threading.Lock()

        def on_event(event: dict[str, Any]) -> None:
            # A cancelled task's log ends before its run has stopped, and refuses the interrupt's last events.
            with appending, suppress(RuntimeError):
                request.trajectory.append(event)

        conversation = conversation_id(request.context_id)
        # Bound before the run starts, so a live trajectory read or a mid-turn snapshot finds the conversation.
        self.set_session_ref_for_context(request.context_id, conversation)
        spent: list[Usage] = []
        async with self._holding(conversation), self._in_workspace():
            try:
                # Reading the file parts is part of the task, and so counts toward its time.
                async with asyncio.timeout(config.timeout_seconds):
                    try:
                        text, images = await self._prompt(request, conversation)
                    except (ValueError, OSError, httpx.HTTPError, httpx.InvalidURL) as exc:
                        return TaskResult.failure("openhands.attachment", f"could not read a file part: {exc}")
                    outcome = await self.runtime.run(Turn(
                        conversation_id=conversation,
                        conversations_dir=self.conversations_dir,
                        workspace=workspace,
                        text=text,
                        images=images,
                        mcp_servers=_with_role(request.mcp_servers, config.role),
                        skills_dir=self.skills_dir,
                        skill_names=tuple(str(skill["name"]) for skill in request.skills),
                        config=config,
                        endpoint=self.endpoint,
                        on_event=on_event,
                        on_usage=spent.append,
                    ))
            except TimeoutError:
                message = f"OpenHands did not finish within {config.timeout_seconds}s"
                timed_out = TaskResult.builder().failed("openhands.timeout", message).add_text(message)
                return (timed_out.usage(spent[-1]) if spent else timed_out).session_ref(conversation).build()
        builder = TaskResult.builder().session_ref(conversation).usage(outcome.usage)
        if outcome.error_code is None:
            return builder.succeeded().add_text(outcome.text).build()
        return builder.failed(outcome.error_code, outcome.text, error_type=outcome.error_type).add_text(outcome.text).build()

    async def _prompt(self, request: TaskRequest[OpenHandsConfig], conversation: str) -> tuple[str, tuple[str, ...]]:
        """The message text and its images as data URIs. Files are kept on disk, in the conversation's own directory,
        for its later turns and its snapshots."""
        files_dir = self.inputs_dir / conversation / request.task_id
        chunks: list[str] = []
        images: list[str] = []
        for index, part in enumerate(request.parts):
            if isinstance(part, TextPart):
                chunks.append(part.text)
            elif isinstance(part, DataPart):
                chunks.append(json.dumps(part.data, indent=2, default=str))
            elif isinstance(part, FilePart):
                files_dir.mkdir(parents=True, exist_ok=True)
                path = await _materialize(part, index, files_dir)
                media_type = _media_type(part)
                if media_type in _IMAGE_TYPES:
                    images.append(f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode()}")
                chunks.append(f"[Attached file ({media_type}) saved at {path}]")
        return "\n\n".join(chunks), tuple(images)

    @asynccontextmanager
    async def _holding(self, conversation: str) -> AsyncIterator[None]:
        entry = self._conversation_locks.setdefault(conversation, _ConversationLock())
        entry.holders += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.holders -= 1
            if not entry.holders:
                del self._conversation_locks[conversation]

    @asynccontextmanager
    async def _in_workspace(self) -> AsyncIterator[None]:
        async with self._workspace_lock:
            self._running += 1
            self._idle.clear()
        try:
            yield
        finally:
            self._running -= 1
            if not self._running:
                self._idle.set()

    def _require_idle(self) -> None:
        if self._running:
            raise HTTPException(status_code=409, detail="a task is running; restore the workspace once it finishes")

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        conversation = conversation_id(request.context_id)
        directory = self.conversations_dir / conversation
        with tempfile.TemporaryDirectory() as scratch:
            # The conversation and the workspace are copied together, so a restore cannot fall between them.
            conversation_archive = Path(scratch) / "conversation.tar.gz"
            workspace_archive = Path(scratch) / "workspace.tar.gz"
            async with self._holding(conversation), self._workspace_lock:
                if not (directory / BASE_STATE).is_file():
                    raise HTTPException(status_code=404, detail=f"No conversation for context '{request.context_id}'")
                await asyncio.to_thread(_pack, directory, conversation_archive, self.inputs_dir / conversation)
                if request.objects.workspace is not None:
                    # Tasks in other contexts work in the same workspace: those running end first, and new ones wait
                    # for the lock this holds.
                    await self._idle.wait()
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(_pack, self.workspace, workspace_archive)
            objects = {"trajectory": (await upload(request.objects.trajectory, conversation_archive)).model_dump(mode="json")}
            if request.objects.workspace is not None:
                objects["workspace"] = (await upload(request.objects.workspace, workspace_archive)).model_dump(mode="json")
        return {"context_id": request.context_id, "objects": objects}

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        context_id = request.target_context_id or uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as scratch:
            conversation_archive = Path(scratch) / "conversation.tar.gz"
            await download(request.objects.trajectory, conversation_archive)
            if request.objects.workspace is not None:
                workspace_archive = Path(scratch) / "workspace.tar.gz"
                await download(request.objects.workspace, workspace_archive)
            async with self._workspace_lock:
                self._require_idle()
                workspace = workspace_archive if request.objects.workspace is not None else None
                try:
                    await asyncio.to_thread(self._restore, conversation_archive, workspace, conversation_id(context_id))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"context_id": context_id}

    def _restore(self, conversation_archive: Path, workspace_archive: Path | None, target: str) -> None:
        """Extract both archives, check the conversation and point it at the target's paths, and only then put each in
        place: one that fails to extract, or holds no conversation, changes nothing. The target gets the attached
        files as they were saved, in its own directory, whatever other contexts have done to theirs."""
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        conversation = _stage_conversation(conversation_archive, self.conversations_dir, target, self.inputs_dir / target)
        staged = [conversation]
        try:
            if workspace_archive is not None:
                self.workspace.mkdir(parents=True, exist_ok=True)
                staged.append(_extract(workspace_archive, self.workspace))
                _replace_contents(self.workspace, staged[-1])
            shutil.rmtree(self.inputs_dir / target, ignore_errors=True)
            if (conversation / INPUTS).is_dir():
                shutil.move(conversation / INPUTS, self.inputs_dir / target)
            shutil.rmtree(self.conversations_dir / target, ignore_errors=True)
            conversation.rename(self.conversations_dir / target)
        finally:
            for path in staged:
                shutil.rmtree(path, ignore_errors=True)

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, Any]:
        skill_dir = self.skills_dir / request.name
        skill_dir.mkdir(parents=True, exist_ok=False)
        (skill_dir / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, Any]:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        staging = self.skills_dir / f".{request.name}.{uuid.uuid4().hex}"
        try:
            # The protocol admits only normalized relative paths, so none leaves the staging directory.
            for item in request.skill_bundle.files:
                await download(item.object, staging / item.path)
            staging.rename(self.skills_dir / request.name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {"name": request.name}
