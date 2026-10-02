"""The reads a capture makes against a running agent's sidecar.

``capture_workspace`` raises on any failure — without the tar there is nothing to
grade — while ``read_partial_trajectory`` never does, because the bundle alone is
what makes a point gradable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from agentenv_protocol.a2a_agent import ObjectSnapshotSaveResponse

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import (
    SNAPSHOT_TRAJECTORY_OBJECT_NAME,
    SNAPSHOT_WORKSPACE_OBJECT_NAME,
    FetchedTrajectory,
    TrajectoryUpload,
    bounded_echo,
    fetch_trajectory,
    invoke_transfer,
    snapshot_save_call,
    trajectory_mode,
)
from agent_env.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceCapture:
    """One ``/ext/snapshot`` call plus the ``FileArtifactUniverse`` wrapping it."""

    universe_id: str
    universe_version: Optional[int]
    # Where the files landed; what restores.
    bundle_object_url: str
    # What was presigned — not always where the agent wrote. Per-service state,
    # captured alongside, still goes here.
    capture_prefix: str


@dataclass
class TrajectoryCapture:
    trajectory: Any = None
    object_url: str | None = None
    # Non-None means this read produced no trajectory; the caller records it as a
    # `partial` capture reason.
    reason: Optional[str] = None


async def capture_workspace(
    *,
    a2a_url: str,
    a2a_card: dict,
    agent_name: str,
    a2a_context_id: str,
    artifact_id: str,
    timeout_seconds: float,
) -> WorkspaceCapture:
    """Tar the agent's workspace to a fresh S3 prefix and wrap it as a universe.

    Raises on any failure — without the tar there is nothing to grade."""
    from agent_env.artifact.store import get_artifact_store

    snapshot_ext = A2AAgent.find_extension(a2a_card or {}, A2AAgent.EXT_SNAPSHOT)
    if not snapshot_ext:
        raise RuntimeError(f"Agent '{agent_name}' does not advertise the snapshot extension")
    save_method, save_path = A2AAgent.operation(snapshot_ext, "save")

    # The random suffix, not the version, isolates captures: two concurrent ones
    # can peek the same version. to_thread because this loop is shared by every
    # concurrent rollout, so a blocking call here stalls their poll loops.
    snapshot_version = await asyncio.to_thread(get_artifact_store().next_version, artifact_id)
    config = get_config()
    capture_key_prefix = (
        f"{config.get_artifact_key_prefix()}agent_snapshots/{artifact_id}/{snapshot_version}-{uuid.uuid4().hex[:8]}/"
    )
    store = config.get_object_store()
    capture_prefix = store.object_url(capture_key_prefix)

    call = await asyncio.to_thread(
        snapshot_save_call,
        save_method,
        store,
        agent_name=agent_name,
        context_id=a2a_context_id,
        capture_prefix=capture_prefix,
    )
    save_body = await invoke_transfer(
        a2a_url + save_path,
        call,
        verb="POST",
        operation="snapshot save",
        timeout=timeout_seconds,
        response_model=ObjectSnapshotSaveResponse,
    )

    if call.mode == "objects":
        # The agent's answer is only a claim: a snapshot missing either object cannot be restored.
        listed = await asyncio.to_thread(store.list_at, capture_prefix)
        stored = {url.rsplit("/", 1)[-1] for url in listed}
        missing = {SNAPSHOT_TRAJECTORY_OBJECT_NAME, SNAPSHOT_WORKSPACE_OBJECT_NAME} - stored
        if missing:
            raise RuntimeError(
                f"snapshot save from agent '{agent_name}' left {sorted(missing)} out of the object store"
            )
        registered_prefix = capture_prefix
    else:
        echoed = save_body.get("s3_prefix")
        if not echoed:
            raise RuntimeError(f"snapshot save response missing 's3_prefix': {save_body}")
        registered_prefix = bounded_echo(capture_prefix, echoed)

    from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse

    # Unserialized — a concurrent version race on one artifact id is
    # `ArtifactStore.put_document`'s retry to absorb.
    universe = await asyncio.to_thread(
        FileArtifactUniverse.put_existing, id=artifact_id, s3_url=registered_prefix
    )
    # `put_existing` always sets it, but the field is Optional on the artifact, and a
    # row carrying None here would read as an ungradable capture rather than an error.
    if not universe.bundle_object_url:
        raise RuntimeError(
            f"FileArtifactUniverse {universe.id} v{universe.version} registered no "
            f"bundle url for {registered_prefix}"
        )
    logger.info(
        "Snapshot captured: agent=%s context_id=%s -> FileArtifactUniverse %s v%s at %s",
        agent_name, a2a_context_id, universe.id, universe.version,
        universe.bundle_object_url,
    )
    return WorkspaceCapture(
        universe_id=universe.id,
        universe_version=universe.version,
        bundle_object_url=universe.bundle_object_url,
        capture_prefix=capture_prefix,
    )


async def read_partial_trajectory(
    *,
    a2a_url: str,
    a2a_card: dict,
    context_id: str,
    timeout_seconds: float,
    trajectory_output_prefix: str,
) -> TrajectoryCapture:
    """Read the trajectory-so-far for an in-progress run.

    Separate from the workspace bundle because not every harness's snapshot
    carries the transcript, which would leave interior points with no trajectory.

    Served by ``get`` in its ``context_id`` mode — cumulative, and keyed on the a2a
    context id each image maps to its own session. An agent advertising ``get`` as
    ``task_id``-only cannot answer this: a task id names one finished turn, which
    404s while the run it belongs to is still going.

    Never raises: a failure yields a ``reason`` the caller records as ``partial``,
    so an agent serving neither shape still yields a usable curve minus the
    trajectory column.
    """
    traj_ext = A2AAgent.find_extension(a2a_card or {}, A2AAgent.EXT_TRAJECTORY)
    if not traj_ext:
        return TrajectoryCapture(reason="trajectory_ext_unavailable")
    # Presence, not truthiness: a card may advertise a method as a bare `{}`,
    # which is falsy. Testing truth would read such a card as not having it.
    get_method, get_path = A2AAgent.operation(traj_ext, "get")
    if get_method is None:
        return TrajectoryCapture(reason="trajectory_get_unadvertised")
    # Only an explicit request contract offers the context mode: a `get` that declares
    # none may be task_id-only, and would 400 on every tick.
    if "request" not in get_method:
        return TrajectoryCapture(reason="trajectory_context_unsupported")
    store = get_config().get_object_store()
    mode = trajectory_mode(get_method, store, by="context_id")
    if mode is None:
        return TrajectoryCapture(reason="trajectory_context_unsupported")

    upload = None
    if mode == "objects":
        try:
            object_url = trajectory_object_url(trajectory_output_prefix, store=store)
            upload = await asyncio.to_thread(TrajectoryUpload.to, store, object_url)
        except Exception as exc:
            logger.warning("partial trajectory grant failed for %s: %s", context_id, exc)
            return TrajectoryCapture(reason="trajectory_grant_unavailable")
    try:
        fetched = await fetch_trajectory(
            a2a_url + get_path, {"context_id": context_id}, upload=upload, timeout=timeout_seconds
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            # No transcript yet — the agent hasn't written a turn.
            return TrajectoryCapture(reason="trajectory_session_missing")
        return TrajectoryCapture(reason=f"trajectory_http_{exc.response.status_code}")
    except Exception as exc:
        logger.warning("partial trajectory read failed for %s: %s", context_id, exc)
        return TrajectoryCapture(reason="trajectory_read_failed")

    if fetched.object_url is not None:
        return TrajectoryCapture(object_url=fetched.object_url)
    if not fetched.inline:
        return TrajectoryCapture(reason="trajectory_empty")
    return TrajectoryCapture(trajectory=fetched.inline)


def upload_trajectory(
    trajectory: Any, trajectory_output_prefix: str, *, name: Optional[str] = None
) -> str:
    """Serialize a trajectory payload and store it, returning its object URL.

    Always uploaded from the worker side, never by the agent: sandbox VMs hold
    static STS env vars with a fixed expiry, and long runs exceed it.

    ``name`` makes the object addressable by an id the caller already records;
    omit it to get a random one, which is what writing repeatedly under a
    single prefix needs, since a fixed name would overwrite the last upload.
    """

    store = get_config().get_object_store()
    # Via the store, not urlparse().path: a prefix naming a different bucket
    # must fail loudly, not silently write to the configured one.
    prefix_key = store.get_object_key(trajectory_output_prefix)
    if prefix_key and not prefix_key.endswith("/"):
        prefix_key += "/"
    key = f"{prefix_key}trajectory-{name or uuid.uuid4().hex[:12]}.json"
    body = json.dumps(trajectory, indent=2, default=str).encode()
    return store.put(key, body, content_type="application/json")


def store_trajectory(
    fetched: FetchedTrajectory, trajectory_output_prefix: str, *, name: Optional[str] = None
) -> str | None:
    """The URL of a fetched trajectory, storing it first when the agent returned it inline."""
    if fetched.object_url is not None:
        return fetched.object_url
    if fetched.legacy_prefix:
        urls = get_config().get_object_store().list_at(fetched.legacy_prefix)
        return urls[0] if urls else None
    if fetched.inline is not None:
        return upload_trajectory(fetched.inline, trajectory_output_prefix, name=name)
    return None


def trajectory_object_url(
    trajectory_output_prefix: str, *, store, name: Optional[str] = None
) -> str:
    """Return the durable URL used for one agent-written trajectory."""
    prefix_key = store.get_object_key(trajectory_output_prefix)
    if prefix_key and not prefix_key.endswith("/"):
        prefix_key += "/"
    key = f"{prefix_key}trajectory-{name or uuid.uuid4().hex[:12]}.json"
    return store.object_url(key)
