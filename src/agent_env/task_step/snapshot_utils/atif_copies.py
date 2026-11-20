"""A run's atif/1 copies: when the run prescribes atif/1, each turn's copy of the conversation is read inline from the
agent once the turn ends and stored as ``<trajectory_id>.json`` in the run's folder for that agent, overwritten each
turn, with the turn's aux log and turn context beside it and the media it names pulled into its ``media/``. It's
checked as a producer's document and the turn records how it checked. The reads run one after another, off the turn's
path, so the next message isn't held for them; a copy that can't be read or stored never fails the run."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import mimetypes
import re
import uuid
from collections.abc import Mapping
from typing import Any

import httpx
from agentenv_protocol.a2a_agent import TrajectoryMediaResponse
from agentenv_protocol.a2a_agent.extensions import MAX_MEDIA_OBJECTS
from agentenv_protocol.atif import (
    FORMAT as ATIF,
    MEDIA_PATH,
    PROFILE_VERSION,
    SCHEMA_VERSION,
    Finding,
    errors_in,
    media_file,
    profile_of,
    validate,
)
from agentenv_protocol.atif.validator import ERROR

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import (
    DEFAULT_TRAJECTORY_MAX_BYTES,
    TRANSFER_TIMEOUT_SECONDS,
    TrajectoryFormats,
    TransferCall,
    invoke_transfer,
    write_object,
)
from agent_env.a2a_agent.staging import transfer_store
from agent_env.config import get_config
from agent_env.store.object_store import ObjectStore
from agent_env.store.object_store.object_store import issues_grants_to
from agent_env.task_step.snapshot_utils.agent_state_capture import trajectory_prefix_key
from agent_env.task_step.context import AtifRecord, DeployedAgent, TaskStepContext

logger = logging.getLogger(__name__)

#: Where a run records its atif/1 folders: ``{"run": <id>, "agents": {<agent>: {"folder", "sessions"}}}``, each
#: session (A2A context) naming the document its last copy was stored as.
ATIF_COPIES_KEY = "atif_copies"
VALID, INVALID, NEWER, UNAVAILABLE = "valid", "invalid", "newer", "unavailable"
#: A trajectory id the worker stores a document under: the agent mints it, so it must be a plain file name.
_DOCUMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_VERSION = re.compile(r"(\d+)\.(\d+)")
#: How much of an invalid copy's first finding its record keeps as the reason; ``rules`` names every rule broken.
_REASON_CHARS = 300
_VALIDATOR_VERSION = importlib.metadata.version("agentenv-framework-protocol")


def atif_folder(context: TaskStepContext, agent_name: str) -> str:
    """The run's folder for ``agent_name``'s atif/1 copies, ``{prefix}atif/instance_id={id}/{agent_name}/``, recorded
    in ``metadata["atif_copies"]``; a run with no instance id gets one ``adhoc-<uuid12>`` for all its agents."""
    copies = context.metadata.setdefault(ATIF_COPIES_KEY, {})
    agent = copies.setdefault("agents", {}).setdefault(agent_name, {"sessions": {}})
    if "folder" not in agent:
        run = copies.setdefault("run", context.instance_id or f"adhoc-{uuid.uuid4().hex[:12]}")
        config = get_config()
        folder = config.get_object_store().object_url(
            f"{config.get_artifact_key_prefix()}atif/instance_id={run}/{agent_name}/")
        agent["folder"] = folder if folder.endswith("/") else f"{folder}/"
    return agent["folder"]


class _Unavailable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class AtifCopies:
    """A step's atif/1 copies of one agent's conversation: ``turn`` starts each ended turn's, and ``records`` waits
    for them all. A run that doesn't prescribe atif/1 records None for each turn; one that prescribes it to an agent
    that doesn't offer it records ``unavailable``, ``format_not_offered``."""

    def __init__(self, context: TaskStepContext, agent: DeployedAgent, a2a_url: str,
                 extension: Mapping[str, Any] | None, *, prescribed: bool) -> None:
        self._context, self._agent, self._a2a_url, self._extension = context, agent, a2a_url, extension
        self._prescribed = prescribed
        self._offered = ATIF in TrajectoryFormats.of(extension).formats
        self._turns: list[asyncio.Task[AtifRecord] | AtifRecord | None] = []
        self._last: asyncio.Task[AtifRecord] | None = None
        #: The media files the folder holds and their sizes (None until read), listed once and kept as this step pulls
        #: more.
        self._held: dict[str, int | None] | None = None

    def turn(self, task_id: str, name: str) -> None:
        """Copy the ended task ``task_id``'s atif/1 document, its aux log and turn context stored under ``name``."""
        if not self._prescribed:
            self._turns.append(None)
        elif not self._offered:
            self._turns.append(_unavailable("format_not_offered"))
        else:
            self._last = asyncio.create_task(self._copy(task_id, name, self._last))
            self._turns.append(self._last)

    async def records(self) -> list[AtifRecord | None]:
        """Each turn's record, once every copy has been stored; cancelled while waiting, it cancels the copies left."""
        try:
            return [await turn if isinstance(turn, asyncio.Task) else turn for turn in self._turns]
        except BaseException:
            self.cancel()
            raise

    def cancel(self) -> None:
        for turn in self._turns:
            if isinstance(turn, asyncio.Task):
                turn.cancel()

    async def _copy(self, task_id: str, name: str, previous: asyncio.Task[AtifRecord] | None) -> AtifRecord:
        if previous is not None:  # each copy overwrites the one before, so they're stored in turn order
            await asyncio.wait([previous])
        try:
            return await self._stored(task_id, name)
        except Exception as exc:
            reason = _reason(exc)
            logger.warning(f"agent '{self._agent.agent_name}': the atif/1 copy of task {task_id} is unavailable "
                           f"({reason}; continuing): {exc}")
            return _unavailable(reason)

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
        url = await asyncio.to_thread(store.put, f"{folder}{trajectory_id}.json", _json(document),
                                      content_type="application/json", allow_overwrite=True)
        aux_url = turn_context_url = None
        if body.get("aux") is not None:
            aux_url = await asyncio.to_thread(store.put, f"{folder}turns/{name}/aux.json", _json(body["aux"]),
                                              content_type="application/json", allow_overwrite=True)
        if body.get("turn_context") is not None:
            turn_context_url = await asyncio.to_thread(
                store.put, f"{folder}turns/{name}/turn_context.json", _json(body["turn_context"]),
                content_type="application/json", allow_overwrite=True)
        root = profile_of(document)
        newer = _newer(document.get("schema_version"), SCHEMA_VERSION) or _newer(root.get("version"), PROFILE_VERSION)
        found = await asyncio.to_thread(validate, document, reader=newer)
        try:
            infos = _media_infos(document)
        except (AttributeError, TypeError):  # a shape no ATIF has, which the validator reports; nothing to walk
            infos = None
        if infos is not None:
            session_id = document.get("session_id")
            if isinstance(session_id, str):
                await self._link(store, folder, session_id, trajectory_id)
            held = await self._pull_media(infos, session_id, store, folder)
            found += _media_findings(infos, held)
        errors = errors_in(found)
        status = NEWER if newer else INVALID if errors else VALID
        if status == INVALID:
            logger.warning(f"agent '{self._agent.agent_name}': the atif/1 copy of task {task_id} is invalid "
                           f"(continuing): {errors[0]}")
        steps = document.get("steps")
        last = steps[-1] if isinstance(steps, list) and steps and isinstance(steps[-1], Mapping) else {}
        adapter = root.get("adapter")
        return AtifRecord(
            url=url, through_step=last.get("step_id") if isinstance(last.get("step_id"), int) else None,
            status=status, rules=sorted({finding.rule for finding in errors}),
            reason=str(errors[0])[:_REASON_CHARS] if errors else None, validator_version=_VALIDATOR_VERSION,
            adapter_version=adapter.get("version") if isinstance(adapter, Mapping) else None,
            aux_url=aux_url, turn_context_url=turn_context_url,
        )

    async def _link(self, store: ObjectStore, folder: str, session_id: str, trajectory_id: str) -> None:
        """Point the session's earlier document, when a restart began this one, to this one: the agent that wrote
        the earlier one may be gone, so the worker sets its ``continued_trajectory_ref``. The session names this one
        only once that's done, so a link that fails is made by a later turn."""
        sessions = self._context.metadata[ATIF_COPIES_KEY]["agents"][self._agent.agent_name]["sessions"]
        earlier = sessions.get(session_id)
        if earlier is not None and earlier != trajectory_id:
            key = f"{folder}{earlier}.json"
            try:
                document = json.loads(await asyncio.to_thread(store.read, key))
                document["continued_trajectory_ref"] = f"{trajectory_id}.json"
                await asyncio.to_thread(store.put, key, _json(document), content_type="application/json",
                                        allow_overwrite=True)
            except Exception as exc:
                logger.warning(f"agent '{self._agent.agent_name}': linking {earlier} to {trajectory_id} failed, so a "
                               f"later turn tries again (continuing): {exc}")
                return
        sessions[session_id] = trajectory_id

    async def _pull_media(self, infos: Mapping[str, Mapping[str, Any]], context_id: Any, store: ObjectStore,
                          folder: str) -> dict[str, int | None]:
        """The media files the folder holds, with their sizes, once the agent has pushed what the document names
        (``infos``, by path) and the folder lacks. A file the agent answers missing or failed for may be stored already, by a push whose answer
        was lost, so the store is asked before it's counted missing; a held file's size is read from the store once."""
        if self._held is None:
            listed = await asyncio.to_thread(store.list, f"{folder}media/")
            self._held = {key[len(folder):]: None for key in listed}
        held = await self._pushed(infos, context_id, store, folder, self._held)
        for path in infos:
            if path in held and held[path] is None:
                metadata = await asyncio.to_thread(store.get_object_metadata, f"{folder}{path}")
                held[path] = None if metadata is None else metadata.size
        return held

    async def _pushed(self, infos: Mapping[str, Mapping[str, Any]], context_id: Any, store: ObjectStore, folder: str,
                      held: dict[str, int | None]) -> dict[str, int | None]:
        """``held``, with what the agent pushed of the files the document names that it lacks."""
        wanted = [path for path in infos if MEDIA_PATH.match(path) and path not in held]
        method, endpoint = A2AAgent.operation(self._extension, "media")
        card, sandbox_type = self._agent.a2a_card, self._agent.sandbox_type
        if not wanted or method is None or endpoint is None or (
                transfer_store(store, self._a2a_url, card, sandbox_type=sandbox_type) is store
                and not issues_grants_to(store, sandbox_type)):
            return held
        cap = store.max_single_upload_bytes
        wanted = [path for path in wanted if cap is None or (infos.get(path, {}).get("bytes") or 0) <= cap]
        unconfirmed: list[str] = []
        for start in range(0, len(wanted), MAX_MEDIA_OBJECTS):
            batch = wanted[start:start + MAX_MEDIA_OBJECTS]
            grants = transfer_store(store, self._a2a_url, card, sandbox_type=sandbox_type)  # one per call
            objects = {
                path: write_object(grants, grants.object_url(f"{folder}{path}"), media_type=_media_type(path, infos),
                                   max_bytes=max(infos.get(path, {}).get("bytes") or DEFAULT_TRAJECTORY_MAX_BYTES, 1))
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


def _media_findings(infos: Mapping[str, Mapping[str, Any]], held: Mapping[str, int | None]) -> list[Finding]:
    """``media-bytes`` for each media file the folder doesn't hold as the profile describes it: missing, a profile
    sha256 other than its name's, or a profile size other than the stored file's. The agent checked each file it pushed
    against its name, before and after the upload, so held files aren't read back."""
    found: list[Finding] = []
    for path, info in infos.items():
        if not MEDIA_PATH.match(path):
            continue
        if path not in held:
            found.append(Finding("media-bytes", ERROR, "", f"{path} isn't stored beside the document"))
            continue
        if info.get("sha256") is not None and info["sha256"] != media_file(path):
            found.append(Finding("media-bytes", ERROR, "", f"{path}'s profile gives sha256 {info['sha256']}"))
        if info.get("bytes") is not None and held[path] is not None and info["bytes"] != held[path]:
            found.append(Finding("media-bytes", ERROR, "",
                                 f"{path}'s profile gives {info['bytes']} bytes, and the stored file has {held[path]}"))
    return found


def _media_infos(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each media file the document names (``media_paths``), by path, with its profile entry and the media type of
    the part that names it when the entry has none."""
    infos: dict[str, dict[str, Any]] = {}
    for trajectory in _trajectories(document):
        for step in trajectory.get("steps") or []:
            for holder in [step, *((step.get("observation") or {}).get("results") or [])]:
                for path, info in (profile_of(holder).get("media") or {}).items():
                    infos.setdefault(path, dict(info) if isinstance(info, Mapping) else {})
                content = holder.get("message") if holder is step else holder.get("content")
                for part in content if isinstance(content, list) else []:
                    source = part.get("source") if isinstance(part, Mapping) else None
                    if isinstance(source, Mapping) and isinstance(source.get("path"), str):
                        info = infos.setdefault(source["path"], {})
                        info.setdefault("media_type", source.get("media_type"))
    return infos


def _trajectories(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [document, *(sub for nested in document.get("subagent_trajectories") or [] for sub in _trajectories(nested))]


def _media_type(path: str, infos: Mapping[str, Mapping[str, Any]]) -> str:
    return infos.get(path, {}).get("media_type") or mimetypes.guess_type(path)[0] or "application/octet-stream"


def _newer(version: Any, ours: str) -> bool:
    got, have = _VERSION.search(str(version or "")), _VERSION.search(ours)
    return got is not None and (int(got[1]), int(got[2])) > (int(have[1]), int(have[2]))


def _json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _reason(exc: Exception) -> str:
    if isinstance(exc, _Unavailable):
        return exc.reason
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            code = (exc.response.json().get("error") or {}).get("code")
        except (ValueError, AttributeError):
            code = None
        return code if isinstance(code, str) else f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "unreachable"
    return type(exc).__name__


def _unavailable(reason: str) -> AtifRecord:
    return AtifRecord(url=None, through_step=None, status=UNAVAILABLE, rules=[], reason=reason, validator_version=None,
                      adapter_version=None, aux_url=None, turn_context_url=None)
