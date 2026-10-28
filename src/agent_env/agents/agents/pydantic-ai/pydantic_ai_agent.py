"""Pydantic AI (https://ai.pydantic.dev) as an AgentEnv A2A agent.

Each task is one run of a Pydantic AI agent with the Pydantic AI Harness coding tools in the workspace. A context maps
to one conversation, its message history persisted on disk, so later tasks in the context continue it. This module is
the A2A side and imports no Pydantic AI; ``pydantic_ai_runtime`` runs the agent.
"""

import asyncio
import base64
import io
import json
import mimetypes
import shutil
import tarfile
import tempfile
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
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
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

PYDANTIC_AI_VERSION: Final[str] = "2.61.0"
TRAJECTORY_FORMAT: Final[str] = "pydantic-ai-messages/v1"
# The model when agent-config sets none, as for a peer that other agents message directly.
DEFAULT_MODEL_ENV: Final[str] = "PYDANTIC_AI_A2A_MODEL"
# The model when DEFAULT_MODEL_ENV is unset too, as the other bundled agents have one.
DEFAULT_MODEL: Final[str] = "anthropic/claude-sonnet-4-6"
A2A_PORT: Final[int] = 8000
INSTALL_DIR: Final[str] = "/opt/pydantic-ai-a2a"
ENV_FILE: Final[str] = "agent.env"
# A conversation's directory holds its message history and, under INPUTS, the files attached in it.
MESSAGES: Final[str] = "messages.json"
INPUTS: Final[str] = "inputs"
# Written into a conversation's archive: the directory it was saved from, under which its messages name attached files.
SOURCE: Final[str] = ".pydantic-ai-a2a-source.json"
FETCH_TIMEOUT_SECONDS: Final[int] = 120
MAX_ATTACHMENT_BYTES: Final[int] = 256 * 1024 * 1024
# A conversation id is derived from its context id, so a context finds its conversation across restarts.
_CONVERSATIONS: Final[uuid.UUID] = uuid.uuid5(uuid.NAMESPACE_URL, "urn:agentenv:agents:pydantic-ai")
# Images go to the model as images as well as to disk; any other file is left on disk for the agent's tools.
IMAGE_TYPES: Final[tuple[str, ...]] = ("image/png", "image/jpeg", "image/gif", "image/webp")
_NO_BODY_STATUSES: Final[frozenset[int]] = frozenset({204, 205, 304})

# sudo only where it needs no password: VM sandboxes have it, a local sandbox runs docker as the user.
_DOCKER: Final[str] = "$(sudo -n true 2>/dev/null && echo sudo) docker"
INSTALL_COMMANDS: Final[list[str]] = [
    f"{_DOCKER} cp {{agent_ctx_tar}} {{container}}:/tmp/pydantic-ai-a2a-ctx.tar.gz",
    f"{_DOCKER} exec {{container}} sh -c 'mkdir -p {INSTALL_DIR} && tar -xf /tmp/pydantic-ai-a2a-ctx.tar.gz"
    f" -C {INSTALL_DIR} && rm /tmp/pydantic-ai-a2a-ctx.tar.gz'",
    f"{_DOCKER} exec {{container}} sh {INSTALL_DIR}/install.sh",
    # The credentials travel on a heredoc into a 0600 file: the installer logs only a command's first line,
    # and neither a command line nor the container's config ever holds them.
    f"{_DOCKER} exec -i {{container}} sh -c 'umask 077 && cat > {INSTALL_DIR}/{ENV_FILE}' <<'PYDANTIC_AI_A2A_ENV'\n"
    "LITELLM_API_KEY={litellm_api_key}\n"
    "LITELLM_BASE_URL={litellm_base_url}\n"
    "LITELLM_EXTRA_HEADERS={litellm_extra_headers}\n"
    # The KEY='VALUE' lines pointing TLS clients at the local transfer CA, which start.sh applies.
    "AGENT_TRUST_ENV={trust_env}\n"
    "PYDANTIC_AI_A2A_ENV",
    f"{_DOCKER} exec -d -e A2A_PORT={{a2a_port}} {{container}} sh {INSTALL_DIR}/start.sh",
]


class PydanticAIConfig(AgentConfig):
    model: str | None = None
    system_prompt: str | None = None
    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    request_limit: int = 500
    timeout_seconds: int = 1800


@dataclass(frozen=True, slots=True)
class Endpoint:
    """The OpenAI-compatible endpoint the model is called through, with the headers every call carries."""

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
class Image:
    path: Path
    media_type: str


@dataclass(frozen=True, slots=True)
class Turn:
    """One task, as the runtime runs it."""

    conversation_dir: Path
    workspace: Path
    text: str
    images: tuple[Image, ...]
    mcp_servers: Mapping[str, Any]
    skills_dir: Path
    skill_names: tuple[str, ...]
    config: PydanticAIConfig
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
        """Run ``turn`` to its end, saving the conversation's messages in ``turn.conversation_dir`` once it succeeds.
        Cancelled, it stops the run and the commands it started, and re-raises."""
        ...

    async def stop(self) -> None:
        """Cancel every turn still running: the server is shutting down."""
        ...

    def check_messages(self, data: bytes) -> None:
        """Raise ValueError unless ``data`` is a conversation's messages as the runtime saves them, so a snapshot that
        holds anything else is refused before it replaces a conversation."""
        ...


async def to_completion(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``function(*args, **kwargs)`` in a thread, run to its end: cancelled meanwhile, this waits for the thread before it
    re-raises, so the locks its caller holds stay held until the files the thread writes are written."""
    work = asyncio.ensure_future(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        await asyncio.wait([work])
        raise


async def _detail_body(_request: Request, exc: HTTPException) -> Response:
    """An ``HTTPException`` as the ``{"detail": ...}`` body agent-env reads errors from."""
    if exc.status_code < 200 or exc.status_code in _NO_BODY_STATUSES:
        return Response(status_code=exc.status_code, headers=exc.headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@asynccontextmanager
async def _turns_stop_with_the_server(app: Starlette) -> AsyncIterator[None]:
    """The server's lifespan: as it shuts down, turns still running are cancelled, which kills the commands they
    started, so none outlives the server."""
    try:
        yield
    finally:
        await app.state.agentenv_a2a.agent.runtime.stop()


def model_from_environ(environ: Mapping[str, str]) -> str:
    """The model a task runs when agent-config sets none: DEFAULT_MODEL_ENV's, else DEFAULT_MODEL."""
    return environ.get(DEFAULT_MODEL_ENV) or DEFAULT_MODEL


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
        # Unpadded, base64's length gives the size it decodes to, so an oversized file is refused before it's decoded.
        if len(part.bytes.rstrip("=")) * 3 // 4 > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"{name or 'file'} is larger than {MAX_ATTACHMENT_BYTES} bytes")
        await to_completion(lambda: destination.write_bytes(base64.b64decode(part.bytes)))
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


def _pack(directory: Path, archive: Path, *, record_source: bool = False, leave_out: str | None = None) -> None:
    """``directory``'s contents, but for its entry named ``leave_out``, as a gzipped tar and, for a conversation, the
    directory it was saved from."""
    with tarfile.open(archive, "w:gz") as tar:
        for child in sorted(directory.iterdir()):
            if child.name != leave_out:
                tar.add(child, arcname=child.name)
        if record_source:
            source = json.dumps({"directory": str(directory)}).encode()
            info = tarfile.TarInfo(SOURCE)
            info.size = len(source)
            tar.addfile(info, io.BytesIO(source))


def _extract(archive: Path, directory: Path) -> Path:
    """The archive extracted into a new staging directory inside ``directory``."""
    staging = directory / f".pydantic-ai-a2a-restore-{uuid.uuid4().hex}"
    # An empty workspace's archive has no members, and extracting it creates nothing.
    staging.mkdir()
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def _replace_contents(directory: Path, staging: Path, *, keep: str | None = None) -> None:
    """Replace ``directory``'s contents with those of ``staging``, a directory inside it, but for its entry named
    ``keep``, which stays as it is."""
    for child in directory.iterdir():
        if child == staging or child.name == keep:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in staging.iterdir():
        if child.name != keep:
            child.rename(directory / child.name)
    shutil.rmtree(staging)


def _stage_conversation(
    archive: Path, conversations_dir: Path, target: Path, check: Callable[[bytes], None]
) -> Path:
    """The archived conversation extracted beside the others, its messages naming the files attached in it under
    ``target``, where it goes, rather than where it was saved from. Raises ValueError for an archive whose messages
    ``check`` refuses."""
    staging = _extract(archive, conversations_dir)
    try:
        check((staging / MESSAGES).read_bytes())
        source = staging / SOURCE
        saved_from = json.loads(source.read_text())["directory"] if source.is_file() else None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"the trajectory object holds no Pydantic AI conversation: {type(exc).__name__}: {exc}") from exc
    if saved_from is not None:
        source.unlink()
        # A path is written as is in a message's text and JSON-escaped in the file.
        before, after = json.dumps(f"{saved_from}/")[1:-1], json.dumps(f"{target}/")[1:-1]
        messages = staging / MESSAGES
        messages.write_text(messages.read_text().replace(before, after))
    return staging


@a2a_agent(
    identity=AgentIdentity(
        name="Pydantic AI",
        description=(
            "A Pydantic AI agent with the Pydantic AI Harness coding tools: reads, edits and runs code in its "
            "workspace with MCP tools and skills."
        ),
        version=PYDANTIC_AI_VERSION,
        input_modes=("text", *IMAGE_TYPES),
    ),
    config=PydanticAIConfig,
    config_description="Configure the Pydantic AI agent: model, effort, system prompt and request limit.",
    lifespan=_turns_stop_with_the_server,
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True, description="The task's Pydantic AI messages, also while it runs."),
        enable(
            SNAPSHOT_V1,
            description=(
                "The trajectory object is the context's persisted Pydantic AI conversation, the workspace object a "
                "gzipped tar."
            ),
        ),
        PEER_AGENTS_V1,
        enable(
            INSTALL_V1,
            install_commands=INSTALL_COMMANDS,
            required_params=[
                "container", "agent_ctx_tar", "a2a_port", "litellm_api_key", "litellm_base_url",
                "litellm_extra_headers", "trust_env",
            ],
            a2a_port=A2A_PORT,
        ),
        TRIGGERS_V1,
    ),
)
class PydanticAIAgent(AgentEnvAgent):
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
        self.home: Path = home or Path.home() / ".pydantic-ai-a2a"
        self.workspace: Path = workspace or Path("/workspace")
        self.conversations_dir: Path = self.home / "conversations"
        self.skills_dir: Path = self.home / "skills"
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

    async def run(self, request: TaskRequest[PydanticAIConfig]) -> TaskResult:
        config = request.config
        if not config.model and self.default_model:
            config = config.model_copy(update={"model": self.default_model})
        if not config.model:
            return TaskResult.failure("pydantic_ai.no_model", "no model is configured; set 'model' through agent-config")
        if self.endpoint is None:
            return TaskResult.failure("pydantic_ai.no_endpoint", "LITELLM_BASE_URL is not set", error_type="infra_error")
        workspace = request.workspace or self.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        request.trajectory.set_format(TRAJECTORY_FORMAT)

        def on_event(event: dict[str, Any]) -> None:
            # A cancelled task's log ends before its run has stopped, and refuses what the run records after that.
            with suppress(RuntimeError):
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
                        return TaskResult.failure("pydantic_ai.attachment", f"could not read a file part: {exc}")
                    outcome = await self.runtime.run(Turn(
                        conversation_dir=self.conversations_dir / conversation,
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
                message = f"Pydantic AI did not finish within {config.timeout_seconds}s"
                timed_out = TaskResult.builder().failed("pydantic_ai.timeout", message).add_text(message)
                return (timed_out.usage(spent[-1]) if spent else timed_out).session_ref(conversation).build()
        builder = TaskResult.builder().session_ref(conversation).usage(outcome.usage)
        if outcome.error_code is None:
            return builder.succeeded().add_text(outcome.text).build()
        return builder.failed(outcome.error_code, outcome.text, error_type=outcome.error_type).add_text(outcome.text).build()

    async def _prompt(self, request: TaskRequest[PydanticAIConfig], conversation: str) -> tuple[str, tuple[Image, ...]]:
        """The message text and its images. Every file is kept on disk, in the conversation's own directory, for its
        later turns and its snapshots, and its path added to the text."""
        files_dir = self.conversations_dir / conversation / INPUTS / request.task_id
        chunks: list[str] = []
        images: list[Image] = []
        for index, part in enumerate(request.parts):
            if isinstance(part, TextPart):
                chunks.append(part.text)
            elif isinstance(part, DataPart):
                chunks.append(json.dumps(part.data, indent=2, default=str))
            elif isinstance(part, FilePart):
                files_dir.mkdir(parents=True, exist_ok=True)
                path = await _materialize(part, index, files_dir)
                media_type = _media_type(part)
                if media_type in IMAGE_TYPES:
                    images.append(Image(path, media_type))
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

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        conversation = conversation_id(request.context_id)
        directory = self.conversations_dir / conversation
        with tempfile.TemporaryDirectory() as scratch:
            # The conversation and the workspace are copied together, so a restore cannot fall between them.
            conversation_archive = Path(scratch) / "conversation.tar.gz"
            workspace_archive = Path(scratch) / "workspace.tar.gz"
            async with self._holding(conversation), self._workspace_lock:
                if not (directory / MESSAGES).is_file():
                    raise HTTPException(status_code=404, detail=f"No conversation for context '{request.context_id}'")
                await to_completion(_pack, directory, conversation_archive, record_source=True)
                if request.objects.workspace is not None:
                    # Tasks in other contexts work in the same workspace: those running end first, and new ones wait
                    # for the lock this holds.
                    await self._idle.wait()
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    await to_completion(_pack, self.workspace, workspace_archive,
                                        leave_out=_entry_holding(self.workspace, self.home))
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
            workspace_archive = None
            if request.objects.workspace is not None:
                workspace_archive = Path(scratch) / "workspace.tar.gz"
                await download(request.objects.workspace, workspace_archive)
            async with self._workspace_lock:
                if self._running:
                    raise HTTPException(status_code=409, detail="a task is running; restore the workspace once it finishes")
                target = self.conversations_dir / conversation_id(context_id)
                try:
                    await to_completion(self._restore, conversation_archive, workspace_archive, target)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"context_id": context_id}

    def _restore(self, conversation_archive: Path, workspace_archive: Path | None, target: Path) -> None:
        """Extract both archives and check the conversation before putting either in place: one that fails to
        extract, or holds no conversation, changes nothing."""
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        conversation = _stage_conversation(conversation_archive, self.conversations_dir, target, self.runtime.check_messages)
        staged = [conversation]
        try:
            if workspace_archive is not None:
                self.workspace.mkdir(parents=True, exist_ok=True)
                staged.append(_extract(workspace_archive, self.workspace))
                _replace_contents(self.workspace, staged[-1], keep=_entry_holding(self.workspace, self.home))
            shutil.rmtree(target, ignore_errors=True)
            conversation.rename(target)
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


def _entry_holding(directory: Path, path: Path) -> str | None:
    """The name of ``directory``'s entry that ``path`` is in, as the agent's home is in a workspace an install in a
    home directory works in; None when ``path`` is outside ``directory``."""
    try:
        return path.resolve().relative_to(directory.resolve()).parts[0]
    except (ValueError, IndexError):
        return None
