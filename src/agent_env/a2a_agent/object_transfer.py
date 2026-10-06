"""How agent-env and an A2A agent move objects: the form each extension call takes, the grants it
carries, the call itself, and the limits and time budget that bound it."""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import httpx
from agentenv_protocol.a2a_agent import (
    BundleSkillRequest,
    ChangelogIncrement,
    ContextObjectTrajectoryRequest,
    ContextTrajectoryRequest,
    InlineSkillRequest,
    NamespaceChangelogEnableRequest,
    ObjectChangelogApplyRequest,
    ObjectChangelogApplyResponse,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    SkillBundle,
    SkillBundleFile,
    SnapshotReadObjects,
    SnapshotWriteObjects,
    TaskObjectTrajectoryRequest,
    TaskTrajectoryRequest,
    TrajectoryObjectsResponse,
    TrajectoryWriteObjects,
    card_request_accepts,
    request_fields,
)
from agentenv_protocol.transfers import ReadObject, WriteNamespaceGrant, WriteObject
from pydantic import BaseModel, ValidationError

from agent_env.a2a_agent.protocol import raise_for_extension_status
from agent_env.a2a_agent.staging import StagedObjectStore, transfer_store
from agent_env.config import get_config
from agent_env.store.base import GrantUnavailableError
from agent_env.store.object_store import DEFAULT_CONTENT_TYPE, ObjectStore, read_url
from agent_env.store.object_store.local.grant_server import unreachable_hint

logger = logging.getLogger(__name__)

# "objects": the call carries grants and the agent moves the bytes. "legacy": the inline forms,
# which carry none.
TransferMode = Literal["objects", "legacy"]
_Response = TypeVar("_Response", bound=BaseModel)
_MEDIA_TYPE = re.compile(r"[!#$&^_.+\-|~0-9a-z]+/[!#$&^_.+\-|~0-9a-z]+")
_INCREMENT_NAME = re.compile(r"^(?P<sequence>[0-9]{6})(?:\.[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_TRANSFER_UNAVAILABLE = "transfer_unavailable"  # the SDK's code for a store it could not reach
_STAGING_UNREACHABLE = (
    "The object store's grants do not reach this agent, so its objects were staged on the agent's own "
    "server, and the agent could not reach them there through its own URL. Its sandbox provider may not let "
    "a sandbox call its own public URL; an object store whose grants reach the agent avoids staging."
)

# The time budget of one transfer, outermost first: a grant (the issuing store's
# grant_lifetime_seconds, 12 hours unless configured) outlives agent-env's wait for the agent's
# answer, which outlasts the SDK's retries of stalled connections
# (agentenv_protocol.transfers.TRANSFER_STALL_BUDGET_SECONDS).
TRANSFER_TIMEOUT_SECONDS = 600  # a call during which the agent moves objects
REPLY_TIMEOUT_SECONDS = 120  # a call during which it moves none


@dataclass(frozen=True)
class ObjectLimits:
    max_objects: int
    max_object_bytes: int
    max_total_bytes: int


_GIB = 1024 * 1024 * 1024
DEFAULT_TRAJECTORY_MAX_BYTES = _GIB
DEFAULT_SNAPSHOT_TRAJECTORY_MAX_BYTES = _GIB
DEFAULT_SNAPSHOT_WORKSPACE_MAX_BYTES = 5 * _GIB
SKILL_BUNDLE_LIMITS = ObjectLimits(max_objects=1_000, max_object_bytes=_GIB, max_total_bytes=_GIB)
CHANGELOG_LIMITS = ObjectLimits(
    max_objects=10_000, max_object_bytes=_GIB, max_total_bytes=10 * _GIB
)

SNAPSHOT_TRAJECTORY_OBJECT_NAME = "trajectory"
SNAPSHOT_WORKSPACE_OBJECT_NAME = "workspace"
# A portable snapshot's objects, each opaque bytes, and the most each may hold.
_SNAPSHOT_OBJECTS = {
    SNAPSHOT_TRAJECTORY_OBJECT_NAME: DEFAULT_SNAPSHOT_TRAJECTORY_MAX_BYTES,
    SNAPSHOT_WORKSPACE_OBJECT_NAME: DEFAULT_SNAPSHOT_WORKSPACE_MAX_BYTES,
}
_OPAQUE = "application/octet-stream"

def _fields(model: type[BaseModel]) -> tuple[str, ...]:
    return request_fields(model).required


def is_portable_snapshot(file_names: Collection[str]) -> bool:
    """Whether a snapshot was captured through the object form, which names its trajectory
    object ``trajectory``; an older capture holds the runtime's own files."""
    return SNAPSHOT_TRAJECTORY_OBJECT_NAME in file_names


def choose_transfer(
    method: Mapping[str, Any] | None,
    *,
    objects: Collection[str] | None = None,
    legacy: Collection[str] | None = None,
    store: ObjectStore,
    sandbox_type: str | None,
) -> TransferMode | None:
    """How one extension call moves its objects, given the fields each form it can take sends.

    Objects need a store that issues grants reaching the agent's sandbox, of ``sandbox_type``
    (None: unknown). A method without a declared request predates variant negotiation and takes
    the legacy form. None: the agent takes neither form.
    """
    if (
        objects is not None
        and _accepts(method, objects)
        and store.supports_transfer_grants
        and store.grants_reach(sandbox_type)
    ):
        return "objects"
    if legacy is not None and (
        method is None or "request" not in method or _accepts(method, legacy)
    ):
        return "legacy"
    return None


def _accepts(method: Mapping[str, Any] | None, fields: Collection[str]) -> bool:
    request = method.get("request") if method is not None else None
    return isinstance(request, Mapping) and card_request_accepts(request, fields)


def _require_object_form(
    method: Mapping[str, Any] | None,
    model: type[BaseModel],
    store: ObjectStore,
    *,
    operation: str,
    sandbox_type: str | None,
) -> None:
    """Refuse ``operation`` unless the agent takes its object form and the store issues grants
    that reach the agent's sandbox, of ``sandbox_type``: agent-env moves objects no other way."""
    if not _accepts(method, _fields(model)):
        raise RuntimeError(f"{operation}: the agent does not advertise the object form")
    if not store.supports_transfer_grants:
        raise RuntimeError(f"{operation}: the object store does not issue transfer grants")
    if not store.grants_reach(sandbox_type):
        raise RuntimeError(
            f"{operation}: the object store's grants do not reach agents on the "
            f"{sandbox_type or 'unknown'!r} sandbox provider"
        )


@dataclass(frozen=True)
class TransferCall:
    """One extension call ready to send: the form it takes and its body."""

    mode: TransferMode
    payload: dict[str, Any]


async def invoke_transfer(
    url: str,
    call: TransferCall,
    *,
    verb: Literal["POST", "PUT"],
    operation: str,
    timeout: float,
    response_model: type[_Response] | None = None,
    store: ObjectStore | None = None,
) -> Any:
    """Send ``call`` and return the agent's answer, validated against ``response_model`` when
    the call carried grants. An error never quotes the answer to such a call, which may echo a
    grant. ``store`` is the store the call's grants came from: when it stages them on the agent,
    the objects the agent reads are pushed first and the ones it writes are in the store on return."""
    staged = store if isinstance(store, StagedObjectStore) and call.mode == "objects" else None
    try:
        if staged is not None:
            await staged.push()
        async with httpx.AsyncClient() as client:
            send = client.post if verb == "POST" else client.put
            resp = await send(url, json=call.payload, timeout=timeout)
        try:
            raise_for_extension_status(resp, operation=operation, include_body=call.mode == "legacy")
        except httpx.HTTPStatusError as exc:
            hint = _STAGING_UNREACHABLE if staged is not None else (
                unreachable_hint(call.payload) if call.mode == "objects" else None
            )
            if hint is None or not str(exc).endswith(f": {_TRANSFER_UNAVAILABLE}"):
                raise
            raise httpx.HTTPStatusError(f"{exc}. {hint}", request=exc.request, response=exc.response) from exc
        body = resp.json()
        if staged is not None:
            await staged.pull()
    finally:
        if staged is not None:
            await staged.release()
    if call.mode == "objects" and response_model is not None:
        return parse_response(response_model, body, operation=operation)
    return body


def parse_response(model: type[_Response], body: Any, *, operation: str) -> _Response:
    """Validate an agent's ``operation`` response, ignoring fields this release does not know."""
    try:
        return model.model_validate(body, extra="ignore")
    except ValidationError as exc:
        raise RuntimeError(f"{operation} returned an invalid object response") from exc


def write_object(store: ObjectStore, url: str, *, media_type: str, max_bytes: int) -> WriteObject:
    """A write grant for one object, promising no more than one upload to the store can create."""
    if store.max_single_upload_bytes is not None:
        max_bytes = min(max_bytes, store.max_single_upload_bytes)
    return WriteObject(
        media_type=media_type,
        max_bytes=max_bytes,
        write=store.issue_write_grant(url, media_type=media_type, max_bytes=max_bytes),
    )


def read_object(
    store: ObjectStore, url: str, *, media_type: str | None = None, max_bytes: int | None = None
) -> ReadObject:
    """A read grant for the stored object at ``url``, bounded by its size unless ``max_bytes``
    is given. ``media_type`` defaults to the stored content type."""
    metadata = store.get_object_metadata_at(url)
    if metadata is None or metadata.size is None:
        raise ValueError(f"object metadata is unavailable for {url}")
    stored_type = (metadata.content_type or "").partition(";")[0].strip().lower()
    if not _MEDIA_TYPE.fullmatch(stored_type):
        stored_type = DEFAULT_CONTENT_TYPE
    return ReadObject(
        media_type=media_type or stored_type,
        max_bytes=max_bytes or max(metadata.size, 1),
        size_bytes=metadata.size,
        read=store.issue_read_grant(url),
    )


def read_objects_under(
    store: ObjectStore,
    prefix: str,
    *,
    limits: ObjectLimits,
    media_type: str | None = None,
    select: Callable[[str], bool] | None = None,
) -> list[tuple[str, ReadObject]]:
    """Read grants for the objects below ``prefix`` that ``select`` keeps, keyed and ordered by
    their path relative to it. The limits count stored bytes, as a namespace uploader does."""
    prefix = prefix.rstrip("/") + "/"
    prefix_key = store.get_object_key(prefix).rstrip("/") + "/"
    listed = sorted(
        (store.get_object_key(url)[len(prefix_key):], url) for url in store.list_at(prefix)
    )
    if select is not None:
        listed = [(path, url) for path, url in listed if select(path)]
    if len(listed) > limits.max_objects:
        raise ValueError(
            f"{len(listed)} objects under {prefix}; the limit is {limits.max_objects}"
        )
    described: list[tuple[str, ReadObject]] = []
    total_bytes = 0
    for path, url in listed:
        descriptor = read_object(store, url, media_type=media_type)
        if descriptor.size_bytes > limits.max_object_bytes:
            raise ValueError(f"{url} exceeds the {limits.max_object_bytes}-byte object limit")
        total_bytes += descriptor.size_bytes
        if total_bytes > limits.max_total_bytes:
            raise ValueError(
                f"objects under {prefix} exceed the {limits.max_total_bytes}-byte limit"
            )
        described.append((path, descriptor))
    return described


def namespace_grant(
    store: ObjectStore, namespace_url: str, *, limits: ObjectLimits, expires_in: int
) -> WriteNamespaceGrant:
    """A grant for uploads below ``namespace_url``: the store signs the upload policy, and the
    uploader holds to the object count and total size."""
    policy = store.issue_upload_policy(
        namespace_url.rstrip("/") + "/",
        max_object_bytes=limits.max_object_bytes,
        expires_in=expires_in,
    )
    return WriteNamespaceGrant(
        root_path=store.get_object_key(namespace_url).rstrip("/"),
        expires_at=policy.expires_at,
        max_objects=limits.max_objects,
        max_object_bytes=limits.max_object_bytes,
        max_total_bytes=limits.max_total_bytes,
        write=policy.write,
    )


@asynccontextmanager
async def readable_parts(
    parts: list[dict],
    *,
    a2a_url: str,
    card: Mapping[str, Any] | None,
    sandbox_type: str | None,
    lasting: int,
) -> AsyncIterator[list[dict]]:
    """``parts`` as the agent at ``a2a_url`` can read them for ``lasting`` seconds: a file part naming an
    object a configured store owns names an HTTPS URL for it instead, from ``read_url`` or else staged on
    the agent for the length of the block. Other parts, and file parts naming anything else, are sent as
    they are. Raises when an owned object can be given no URL the agent can read."""
    readable = list(parts)
    staged: dict[int, StagedObjectStore] = {}
    for index, part in enumerate(parts):
        file = part.get("file") if part.get("kind") == "file" else None
        uri = file.get("uri") if isinstance(file, dict) else None
        if not isinstance(uri, str):
            continue
        store = get_config().get_object_store_at(uri)
        if not store.owns(uri):
            continue
        url = await asyncio.to_thread(read_url, store, uri, sandbox_type=sandbox_type, lasting=lasting)
        if url is None:
            staging = staged.get(id(store)) or transfer_store(store, a2a_url, card, sandbox_type=sandbox_type)
            if isinstance(staging, StagedObjectStore):
                staged[id(store)] = staging
                url = str(staging.issue_read_grant(uri, expires_in=lasting).url)
        if url is None:
            raise RuntimeError(
                f"{uri} cannot be sent to the agent: {type(store).__name__} neither issues grants that reach "
                f"agents on the {sandbox_type or 'unknown'!r} sandbox provider nor signs URLs, and the agent "
                "serves no staging to send it through"
            )
        readable[index] = {**part, "file": {**file, "uri": url}}
    try:
        for staging in staged.values():
            await staging.push()
        yield readable
    finally:
        for staging in staged.values():
            await staging.release()


def skill_bundle_request(
    store: ObjectStore, *, name: str, description: str, object_url: str
) -> BundleSkillRequest:
    """A portable skill bundle of every object under ``object_url``."""
    described = read_objects_under(store, object_url, limits=SKILL_BUNDLE_LIMITS)
    if not described:
        raise ValueError(f"no objects under {object_url} to send as a skill bundle")
    files = [SkillBundleFile(path=path, object=descriptor) for path, descriptor in described]
    return BundleSkillRequest(
        name=name,
        description=description,
        skill_bundle=SkillBundle(
            max_total_bytes=sum(file.object.max_bytes for file in files), files=files
        ),
    )


def skill_add_call(
    method: Mapping[str, Any] | None,
    store: ObjectStore,
    *,
    name: str,
    description: str,
    skill_md: str | None = None,
    object_url: str | None = None,
    sandbox_type: str | None,
) -> TransferCall:
    """The skill ``add`` call for a skill given as SKILL.md text, sent inline, or as the objects
    under ``object_url``, sent as a bundle of read grants."""
    base = {"name": name, "description": description}
    if object_url is not None:
        _require_object_form(
            method, BundleSkillRequest, store, operation="skill add", sandbox_type=sandbox_type
        )
        request = skill_bundle_request(
            store, name=name, description=description, object_url=object_url
        )
        return TransferCall("objects", request.model_dump(mode="json"))
    if skill_md is not None:
        if choose_transfer(
            method, legacy=_fields(InlineSkillRequest), store=store, sandbox_type=sandbox_type
        ) is None:
            raise RuntimeError("Agent does not advertise the inline skill variant")
        return TransferCall("legacy", {**base, "skill_md": skill_md})
    return TransferCall("legacy", base)


def snapshot_save_call(
    method: Mapping[str, Any] | None,
    store: ObjectStore,
    *,
    agent_name: str,
    context_id: str,
    capture_prefix: str,
    sandbox_type: str | None,
) -> TransferCall:
    """The snapshot ``save`` call that writes one capture below ``capture_prefix``."""
    _require_object_form(
        method,
        ObjectSnapshotSaveRequest,
        store,
        operation=f"snapshot save on agent '{agent_name}'",
        sandbox_type=sandbox_type,
    )
    objects = {
        name: write_object(store, url, media_type=_OPAQUE, max_bytes=max_bytes)
        for name, url, max_bytes in _snapshot_objects(store, capture_prefix)
    }
    request = ObjectSnapshotSaveRequest(
        context_id=context_id, objects=SnapshotWriteObjects(**objects)
    )
    return TransferCall("objects", request.model_dump(mode="json"))


def snapshot_load_call(
    method: Mapping[str, Any] | None,
    store: ObjectStore,
    *,
    agent_name: str,
    bundle_url: str,
    file_names: Collection[str],
    target_context_id: str | None,
    sandbox_type: str | None,
) -> TransferCall:
    """The snapshot ``load`` call that restores the capture at ``bundle_url``. Only a portable
    capture can be restored; an older one holds the runtime's own files."""
    operation = f"snapshot load on agent '{agent_name}'"
    if not is_portable_snapshot(file_names):
        raise RuntimeError(
            f"{operation}: {bundle_url} is not a portable snapshot, so it cannot be restored"
        )
    _require_object_form(
        method, ObjectSnapshotLoadRequest, store, operation=operation, sandbox_type=sandbox_type
    )
    objects = {
        name: read_object(store, url, media_type=_OPAQUE, max_bytes=max_bytes)
        for name, url, max_bytes in _snapshot_objects(store, bundle_url)
        if name in file_names
    }
    request = ObjectSnapshotLoadRequest(
        objects=SnapshotReadObjects(**objects), target_context_id=target_context_id
    )
    return TransferCall("objects", request.model_dump(mode="json", exclude_none=True))


def _snapshot_objects(store: ObjectStore, prefix_url: str) -> list[tuple[str, str, int]]:
    prefix_key = store.get_object_key(prefix_url).rstrip("/") + "/"
    return [
        (name, store.object_url(prefix_key + name), max_bytes)
        for name, max_bytes in _SNAPSHOT_OBJECTS.items()
    ]


def changelog_enable_call(
    method: Mapping[str, Any] | None,
    store: ObjectStore,
    *,
    agent_name: str,
    namespace_url: str,
    expires_in: int,
    sandbox_type: str | None,
) -> TransferCall:
    """The ``enable-changelog`` call that captures below ``namespace_url``. Raises
    GrantUnavailableError when the store cannot sign a namespace grant that lasts ``expires_in``."""
    operation = f"changelog enable on agent '{agent_name}'"
    _require_object_form(
        method,
        NamespaceChangelogEnableRequest,
        store,
        operation=operation,
        sandbox_type=sandbox_type,
    )
    try:
        grant = namespace_grant(
            store, namespace_url, limits=CHANGELOG_LIMITS, expires_in=expires_in
        )
    except GrantUnavailableError as exc:
        raise GrantUnavailableError(
            f"{operation}: the object store cannot issue its namespace grant: {exc}"
        ) from exc
    request = NamespaceChangelogEnableRequest(write_namespace=grant)
    return TransferCall("objects", request.model_dump(mode="json", exclude_none=True))


def changelog_apply_call(
    method: Mapping[str, Any] | None,
    store: ObjectStore,
    *,
    agent_name: str,
    source_url: str,
    up_to_tool_call_exclusive: int | None = None,
    resume_conversation: bool = False,
    target_context_id: str | None = None,
    sandbox_type: str | None,
) -> TransferCall:
    """The ``apply-changelog`` call that replays the capture at ``source_url``: read grants for
    its increments before the cutoff, in sequence order, and none when the cutoff precedes the
    first tool call."""
    _require_object_form(
        method,
        ObjectChangelogApplyRequest,
        store,
        operation=f"changelog apply on agent '{agent_name}'",
        sandbox_type=sandbox_type,
    )
    cutoff = math.inf if up_to_tool_call_exclusive is None else up_to_tool_call_exclusive
    described = read_objects_under(
        store,
        source_url,
        limits=CHANGELOG_LIMITS,
        media_type=_OPAQUE,
        select=lambda path: _increment_sequence(path) < cutoff,
    )
    request = ObjectChangelogApplyRequest(
        increments=[
            ChangelogIncrement(sequence=_increment_sequence(path), object=descriptor)
            for path, descriptor in described
        ],
        resume_conversation=resume_conversation,
        target_context_id=target_context_id,
    )
    return TransferCall("objects", request.model_dump(mode="json", exclude_none=True))


def check_changelog_applied(
    result: ObjectChangelogApplyResponse, call: TransferCall, *, agent_name: str
) -> None:
    """Refuse an ``apply-changelog`` answer that does not account for every increment sent."""
    sent = len(call.payload["increments"])
    if result.count != sent:
        raise RuntimeError(
            f"changelog apply on agent '{agent_name}' reported {result.count} increments "
            f"applied; {sent} were sent"
        )


def _increment_sequence(path: str) -> int:
    match = _INCREMENT_NAME.fullmatch(path)
    if match is None:
        raise ValueError("portable changelog objects must use zero-padded sequence names")
    return int(match["sequence"])


_TRAJECTORY_FORMS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "task_id": (TaskObjectTrajectoryRequest, TaskTrajectoryRequest),
    "context_id": (ContextObjectTrajectoryRequest, ContextTrajectoryRequest),
}


def trajectory_mode(
    method: Mapping[str, Any] | None,
    store: ObjectStore,
    *,
    by: Literal["task_id", "context_id"],
    sandbox_type: str | None,
) -> TransferMode | None:
    """How a trajectory ``get`` selecting by ``by`` moves the trajectory: uploaded by the agent
    through a grant, or returned inline."""
    objects_model, inline_model = _TRAJECTORY_FORMS[by]
    return choose_transfer(
        method,
        objects=_fields(objects_model),
        legacy=_fields(inline_model),
        store=store,
        sandbox_type=sandbox_type,
    )


@dataclass(frozen=True)
class TrajectoryUpload:
    """Where the agent uploads a trajectory, and the grant it uploads with."""

    object_url: str
    write: WriteObject

    @classmethod
    def to(cls, store: ObjectStore, object_url: str) -> TrajectoryUpload:
        write = write_object(
            store,
            object_url,
            media_type="application/json",
            max_bytes=DEFAULT_TRAJECTORY_MAX_BYTES,
        )
        return cls(object_url, write)


@dataclass(frozen=True)
class FetchedTrajectory:
    """Where a trajectory ``get`` left the trajectory: uploaded by the agent through the grant
    (``object_url``) or in the answer (``inline``). Both None: the agent returned none."""

    object_url: str | None = None
    inline: Any = None


async def fetch_trajectory(
    endpoint: str,
    selector: Mapping[str, str],
    *,
    upload: TrajectoryUpload | None = None,
    timeout: float | None = None,
    store: ObjectStore | None = None,
) -> FetchedTrajectory:
    """POST one trajectory ``get``. With ``upload`` the agent uploads the trajectory through its
    grant, issued by ``store``, and the answer is checked. The default wait covers that upload."""
    payload: dict[str, Any] = dict(selector)
    if upload is not None:
        objects = TrajectoryWriteObjects(trajectory=upload.write)
        payload["objects"] = objects.model_dump(mode="json")
    call = TransferCall("objects" if upload is not None else "legacy", payload)
    if timeout is None:
        timeout = TRANSFER_TIMEOUT_SECONDS if upload is not None else REPLY_TIMEOUT_SECONDS
    body = await invoke_transfer(
        endpoint,
        call,
        verb="POST",
        operation="trajectory get",
        timeout=timeout,
        response_model=TrajectoryObjectsResponse,
        store=store,
    )
    if upload is not None:
        return FetchedTrajectory(object_url=upload.object_url)
    if not isinstance(body, Mapping):
        return FetchedTrajectory()
    if body.get("trajectory") is None and body.get("trajectory_s3_prefix"):
        raise RuntimeError(
            "trajectory get answered with trajectory_s3_prefix, which agent-env does not read; "
            "the agent must return the trajectory inline or take the object form"
        )
    return FetchedTrajectory(inline=body.get("trajectory"))
