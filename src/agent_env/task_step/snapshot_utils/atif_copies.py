"""A run's atif/1 copies: when the run prescribes atif/1 to an agent that offers it, each turn's copy of the
conversation is the turn's trajectory. It's read inline from the agent once the turn ends and stored as
``<trajectory_id>.json`` in the run's folder for that agent, overwritten each turn, with the turn's aux log and turn
context beside it and the media it names pulled into its ``media/``; it's checked as a producer's document and the turn
records how it checked. The reads run one after another, off the turn's path, so the next message isn't held for them.
A copy that can't be had never fails the run: that turn's native trajectory is read in its place."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import mimetypes
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Optional

import httpx
from agentenv_protocol.a2a_agent import TrajectoryMediaResponse
from agentenv_protocol.a2a_agent.extensions import MAX_MEDIA_OBJECTS
from agentenv_protocol.atif import (
    FORMAT as ATIF,
    MEDIA_PATH,
    PROFILE_VERSION,
    SCHEMA_VERSION,
    MediaRef,
    StoredMedia,
    errors_in,
    media_refs,
    profile_of,
    validate,
    validate_media,
)

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import (
    DEFAULT_TRAJECTORY_MAX_BYTES,
    TRANSFER_TIMEOUT_SECONDS,
    TrajectoryFormats,
    TransferCall,
    invoke_transfer,
    write_object,
)
from agent_env.a2a_agent.protocol import extension_error_code
from agent_env.a2a_agent.staging import transfer_store
from agent_env.config import get_config
from agent_env.store.object_store import ObjectStore
from agent_env.store.object_store.object_store import issues_grants_to
from agent_env.task_step.context import (
    AtifAgentCopies,
    AtifCopiesMetadata,
    AtifRecord,
    AtifStatus,
    DeployedAgent,
    TaskStepContext,
    TurnTrajectory,
)
from agent_env.task_step.snapshot_utils.agent_state_capture import trajectory_prefix_key

logger = logging.getLogger(__name__)

#: Reads a turn's native trajectory: where it was stored, None when the agent returned none, and its format.
NativeRead = Callable[[], Awaitable[tuple[Optional[str], Optional[str]]]]
#: Where a run records its atif/1 folders (``AtifCopiesMetadata``).
ATIF_COPIES_KEY = "atif_copies"
VALID: AtifStatus = "valid"
INVALID: AtifStatus = "invalid"
NEWER: AtifStatus = "newer"
UNAVAILABLE: AtifStatus = "unavailable"
#: A trajectory id the worker stores a document under: the agent mints it, so it must be a plain file name.
_DOCUMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_VERSION = re.compile(r"(\d+)\.(\d+)")
#: How much of an invalid copy's first finding its record keeps as the reason; ``rules`` names every rule broken.
_REASON_CHARS = 300
_VALIDATOR_VERSION = importlib.metadata.version("agentenv-framework-protocol")


def atif_folder(context: TaskStepContext, agent_name: str) -> str:
    """The run's folder for ``agent_name``'s atif/1 copies, ``{prefix}atif/instance_id={id}/{agent_name}/``, recorded
    in ``metadata["atif_copies"]``; a run with no instance id gets one ``adhoc-<uuid12>`` for all its agents."""
    return _copies_of(context, agent_name)["folder"]


def _copies_of(context: TaskStepContext, agent_name: str) -> AtifAgentCopies:
    copies: AtifCopiesMetadata = context.metadata.setdefault(ATIF_COPIES_KEY, AtifCopiesMetadata(
        run=context.instance_id or f"adhoc-{uuid.uuid4().hex[:12]}", agents={}))
    if agent_name not in copies["agents"]:
        config = get_config()
        folder = config.get_object_store().object_url(
            f"{config.get_artifact_key_prefix()}atif/instance_id={copies['run']}/{agent_name}/")
        copies["agents"][agent_name] = AtifAgentCopies(folder=folder if folder.endswith("/") else f"{folder}/",
                                                      sessions={})
    return copies["agents"][agent_name]


class _Unavailable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class AtifCopies:
    """A step's atif/1 copies of one agent's conversation. When the run prescribes atif/1 to an agent that offers it
    (``replaces_native``), each turn's copy is the turn's trajectory: ``turn`` starts reading it once the turn ends, a
    turn whose copy can't be had has its native trajectory read at once instead (the requested format is best effort),
    and ``join`` fills each turn's record and gives each turn's url. Otherwise the step reads native as ever, and a
    turn's record has None when the run doesn't prescribe atif/1, or ``unavailable`` with ``format_not_offered`` when
    the agent doesn't offer it."""

    def __init__(self, context: TaskStepContext, agent: DeployedAgent, a2a_url: str,
                 extension: Mapping[str, Any] | None, *, prescribed: bool) -> None:
        self._context, self._agent, self._a2a_url, self._extension = context, agent, a2a_url, extension
        self._prescribed = prescribed
        self._offered = ATIF in TrajectoryFormats.of(extension).formats
        self.replaces_native = prescribed and self._offered
        self._pending: list[tuple[int, asyncio.Task[str | None]]] = []
        #: The media files the folder holds and their sizes (None until read), listed once and kept as this step pulls
        #: more.
        self._held: dict[str, int | None] | None = None

    def turn(self, index: int, task_id: str, name: str, record: TurnTrajectory, *, native: NativeRead) -> None:
        """Turn ``index``'s trajectory, its ended A2A task ``task_id``: into ``record`` and, through ``join``, its url,
        the copy's aux log and turn context stored under ``name``. ``native`` reads the turn's native trajectory."""
        if not self._prescribed:
            record["atif"] = None
        elif not self._offered:
            record["atif"] = _unavailable("format_not_offered", task_id)
        else:
            previous = self._pending[-1][1] if self._pending else None
            self._pending.append((index, asyncio.create_task(self._turn(task_id, name, record, native, previous))))

    async def join(self) -> dict[int, str | None]:
        """Each copied turn's url, by its index, once every copy has been stored and its record filled; cancelled while
        waiting, it cancels the copies left."""
        try:
            return {index: await job for index, job in self._pending}
        except BaseException:
            self.cancel()
            raise

    def cancel(self) -> None:
        for _, job in self._pending:
            job.cancel()

    async def _turn(self, task_id: str, name: str, record: TurnTrajectory, native: NativeRead,
                    previous: asyncio.Task[str | None] | None) -> str | None:
        if previous is not None:  # each copy overwrites the one before, so they're stored in turn order
            await asyncio.wait([previous])
        record["atif"] = atif = await self._copy(task_id, name)
        if atif["status"] != UNAVAILABLE:
            record["format"] = ATIF
            return atif["url"]
        try:  # at once, while the agent still holds it
            url, record["format"] = await native()
        except Exception as exc:
            logger.warning(f"agent '{self._agent.agent_name}': task {task_id}'s native trajectory, read in place of its "
                           f"atif/1 copy, failed too (continuing): {exc}")
            return None
        return url

    async def _copy(self, task_id: str, name: str) -> AtifRecord:
        try:
            return await self._stored(task_id, name)
        except Exception as exc:
            reason = _reason(exc)
            logger.warning(f"agent '{self._agent.agent_name}': the atif/1 copy of task {task_id} is unavailable "
                           f"({reason}; continuing): {exc}")
            return _unavailable(reason, task_id)

    async def _stored(self, task_id: str, name: str) -> AtifRecord:
        _, path = A2AAgent.operation(self._extension, "get")
        body = await invoke_transfer(
            self._a2a_url + path, TransferCall("legacy", {"task_id": task_id, "format": ATIF}), verb="POST",
            operation="atif/1 trajectory get", timeout=TRANSFER_TIMEOUT_SECONDS, headers=self._agent.control_headers)
        document = body.get("trajectory") if isinstance(body, Mapping) else None
        if not isinstance(document, Mapping):
            raise _Unavailable("no_document")
        trajectory_id = document.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not _DOCUMENT_ID.fullmatch(trajectory_id):
            raise _Unavailable("invalid_trajectory_id")
        store = get_config().get_object_store()
        folder = trajectory_prefix_key(store, atif_folder(self._context, self._agent.agent_name))
        url = await _put(store, f"{folder}{trajectory_id}.json", document)
        aux_url = None if body.get("aux") is None else await _put(store, f"{folder}turns/{name}/aux.json", body["aux"])
        turn_context_url = None if body.get("turn_context") is None else await _put(
            store, f"{folder}turns/{name}/turn_context.json", body["turn_context"])
        root = profile_of(document)
        newer = _newer(document.get("schema_version"), SCHEMA_VERSION) or _newer(root.get("version"), PROFILE_VERSION)
        try:
            refs = _first_refs(document)
        except (AttributeError, TypeError):  # a shape no ATIF has, which the validator reports; nothing to walk
            refs = None
        media = None
        if refs is not None:
            session_id = document.get("session_id")
            if isinstance(session_id, str):
                await self._link(store, folder, session_id, trajectory_id)
            media = StoredMedia(await self._pull_media(refs, session_id, store, folder))
        found = await asyncio.to_thread(validate, document, reader=newer)
        if media is not None:  # on its own, since validate stops at a document newer than its models
            found += await asyncio.to_thread(validate_media, document, media, reader=newer)
        errors = errors_in(found)
        status = NEWER if newer else INVALID if errors else VALID
        if status == INVALID:
            logger.warning(f"agent '{self._agent.agent_name}': the atif/1 copy of task {task_id} is invalid "
                           f"(continuing): {errors[0]}")
        steps = document.get("steps")
        last = steps[-1] if isinstance(steps, list) and steps and isinstance(steps[-1], Mapping) else {}
        adapter = root.get("adapter")
        return AtifRecord(
            url=url, task_id=task_id, through_step=last.get("step_id") if isinstance(last.get("step_id"), int) else None,
            status=status, rules=sorted({finding.rule for finding in errors}),
            reason=str(errors[0])[:_REASON_CHARS] if errors else None, validator_version=_VALIDATOR_VERSION,
            adapter_version=adapter.get("version") if isinstance(adapter, Mapping) else None,
            aux_url=aux_url, turn_context_url=turn_context_url,
        )

    async def _link(self, store: ObjectStore, folder: str, session_id: str, trajectory_id: str) -> None:
        """Point the session's earlier document, when a restart began this one, to this one: the agent that wrote
        the earlier one may be gone, so the worker sets its ``continued_trajectory_ref``. The session names this one
        only once that's done, so a link that fails is made by a later turn."""
        sessions = _copies_of(self._context, self._agent.agent_name)["sessions"]
        earlier = sessions.get(session_id)
        if earlier is not None and earlier != trajectory_id:
            key = f"{folder}{earlier}.json"
            try:
                document = json.loads(await asyncio.to_thread(store.read, key))
                document["continued_trajectory_ref"] = f"{trajectory_id}.json"
                await _put(store, key, document)
            except Exception as exc:
                logger.warning(f"agent '{self._agent.agent_name}': linking {earlier} to {trajectory_id} failed, so a "
                               f"later turn tries again (continuing): {exc}")
                return
        sessions[session_id] = trajectory_id

    async def _pull_media(self, refs: Mapping[str, MediaRef], context_id: Any, store: ObjectStore,
                          folder: str) -> dict[str, int | None]:
        """The media files the folder holds, with their sizes, once the agent has pushed what the document names
        (``refs``) and the folder lacks. A file the agent answers missing or failed for may be stored already, by a
        push whose answer was lost, so the store is asked before it's counted missing; a held file's size is read from
        the store once."""
        if self._held is None:
            listed = await asyncio.to_thread(store.list, f"{folder}media/")
            self._held = {key[len(folder):]: None for key in listed}
        held = await self._pushed(refs, context_id, store, folder, self._held)
        for path in refs:
            if path in held and held[path] is None:
                metadata = await asyncio.to_thread(store.get_object_metadata, f"{folder}{path}")
                held[path] = None if metadata is None else metadata.size
        return held

    async def _pushed(self, refs: Mapping[str, MediaRef], context_id: Any, store: ObjectStore, folder: str,
                      held: dict[str, int | None]) -> dict[str, int | None]:
        """``held``, with what the agent pushed of the files the document names that it lacks."""
        wanted = [path for path in refs if MEDIA_PATH.match(path) and path not in held]
        method, endpoint = A2AAgent.operation(self._extension, "media")
        card, sandbox_type = self._agent.a2a_card, self._agent.sandbox_type
        if not wanted or method is None or endpoint is None or (
                transfer_store(store, self._a2a_url, card, sandbox_type=sandbox_type) is store
                and not issues_grants_to(store, sandbox_type)):
            return held
        cap = store.max_single_upload_bytes
        wanted = [path for path in wanted if cap is None or (refs[path].info.get("bytes") or 0) <= cap]
        unconfirmed: list[str] = []
        for start in range(0, len(wanted), MAX_MEDIA_OBJECTS):
            batch = wanted[start:start + MAX_MEDIA_OBJECTS]
            grants = transfer_store(store, self._a2a_url, card, sandbox_type=sandbox_type)  # one per call
            objects = {
                path: write_object(grants, grants.object_url(f"{folder}{path}"), media_type=_media_type(refs[path]),
                                   max_bytes=max(refs[path].info.get("bytes") or DEFAULT_TRAJECTORY_MAX_BYTES, 1))
                .model_dump(mode="json")
                for path in batch
            }
            try:
                pushed = await invoke_transfer(
                    self._a2a_url + endpoint, TransferCall("objects", {"context_id": context_id, "objects": objects}),
                    verb="POST", operation="trajectory media", timeout=TRANSFER_TIMEOUT_SECONDS,
                    response_model=TrajectoryMediaResponse, store=grants, headers=self._agent.control_headers)
            except Exception as exc:
                logger.warning(f"agent '{self._agent.agent_name}': the media push of {len(batch)} files failed "
                               f"(continuing): {exc}")
                unconfirmed += batch
                continue
            held.update({path: uploaded.size_bytes for path, uploaded in pushed.objects.items()})
            unconfirmed += [*pushed.missing, *pushed.failed]
        for path in unconfirmed:
            metadata = await asyncio.to_thread(store.get_object_metadata, f"{folder}{path}")
            if metadata is not None:
                held[path] = metadata.size
        return held


def _first_refs(document: Mapping[str, Any]) -> dict[str, MediaRef]:
    """Each media file the document names, by path, as the first place names it."""
    refs: dict[str, MediaRef] = {}
    for ref in media_refs(document):
        refs.setdefault(ref.path, ref)
    return refs


def _media_type(ref: MediaRef) -> str:
    return ref.media_type or mimetypes.guess_type(ref.path)[0] or "application/octet-stream"


def _newer(version: Any, ours: str) -> bool:
    got, have = _VERSION.search(str(version or "")), _VERSION.search(ours)
    return got is not None and (int(got[1]), int(got[2])) > (int(have[1]), int(have[2]))


async def _put(store: ObjectStore, key: str, value: Any) -> str:
    body = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return await asyncio.to_thread(store.put, key, body, content_type="application/json", allow_overwrite=True)


def _reason(exc: Exception) -> str:
    if isinstance(exc, _Unavailable):
        return exc.reason
    if isinstance(exc, httpx.HTTPStatusError):
        return extension_error_code(exc.response) or f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "unreachable"
    return type(exc).__name__


def _unavailable(reason: str, task_id: str) -> AtifRecord:
    return AtifRecord(url=None, task_id=task_id, through_step=None, status=UNAVAILABLE, rules=[], reason=reason,
                      validator_version=None, adapter_version=None, aux_url=None, turn_context_url=None)
