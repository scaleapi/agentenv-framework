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

from agent_env.a2a_agent import A2AAgent

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_ENDPOINT = "/ext/snapshot"
DEFAULT_TRAJECTORY_ENDPOINT = "/ext/trajectory"


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
    # Non-None means this read produced no trajectory; the caller records it as a
    # `partial` capture reason.
    reason: Optional[str] = None


def _bounded_echo(issued: str, echoed: str) -> str:
    """Where the agent says it wrote, clamped to the prefix we issued.

    Sidecars uploading with their own client nest under it and echo that; registering
    the issued prefix instead points the artifact a level too high, which
    `put_existing`'s recursive list hides until a restore cannot find its files.

    Escaping is still refused — `put_existing` registers everything under what it is
    handed, with the worker's credentials. Nesting is free to allow: the presigned
    POST's `starts-with $key` already admits any depth under the issued prefix.
    """
    issued_norm, echoed_norm = issued.rstrip("/"), echoed.rstrip("/")
    if echoed_norm == issued_norm:
        return issued
    if echoed_norm.startswith(issued_norm + "/"):
        return echoed
    logger.warning(
        "snapshot save echoed a prefix outside the one issued; registering the "
        "issued one (issued=%s returned=%s)", issued, echoed,
    )
    return issued


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
    from agent_env.config import get_config

    snapshot_ext = A2AAgent.find_extension(a2a_card or {}, A2AAgent.EXT_SNAPSHOT)
    if not snapshot_ext:
        raise RuntimeError(f"Agent '{agent_name}' does not advertise the snapshot extension")
    ext_params = snapshot_ext.get("params") or {}
    save_url = a2a_url + ext_params.get("endpoint", DEFAULT_SNAPSHOT_ENDPOINT)

    # The random suffix, not the version, isolates captures: two concurrent ones
    # can peek the same version. to_thread because this loop is shared by every
    # concurrent rollout, so a blocking call here stalls their poll loops.
    snapshot_version = await asyncio.to_thread(get_artifact_store().next_version, artifact_id)
    capture_key_prefix = (
        f"agent_snapshots/{artifact_id}/{snapshot_version}-{uuid.uuid4().hex[:8]}/"
    )
    store = get_config().get_object_store()
    capture_prefix = store.object_url(capture_key_prefix)

    # Signed with the step's fresh creds: the sidecar's are the deployer's STS
    # session, frozen at deploy and expired on a long run.
    presigned_post = await asyncio.to_thread(store.signed_post, capture_prefix)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            save_url,
            json={
                "context_id": a2a_context_id,
                "s3_prefix": capture_prefix,
                # Absent, not null, when the backend cannot sign: the sidecar then
                # uploads with its own client.
                **({"presigned_post": presigned_post} if presigned_post else {}),
            },
            timeout=timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"snapshot save failed: {resp.status_code} {resp.text}")
        save_body = resp.json()

    echoed = save_body.get("s3_prefix")
    if not echoed:
        raise RuntimeError(f"snapshot save response missing 's3_prefix': {save_body}")
    registered_prefix = _bounded_echo(capture_prefix, echoed)

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


def _accepts_context_id(method: dict) -> bool:
    """Does this advertised ``get`` take ``context_id``, per its own request contract?

    The keys are mutually exclusive and advertised as a ``oneOf``, so the accepted
    set unions every branch's ``required`` with the flat ``required``/``optional``.
    Distinguishes a card advertising ``get`` as ``task_id``-only — which would 400
    on every tick — without a round-trip.
    """
    request = (method or {}).get("request") or {}
    accepted: set[str] = set()
    for branch in request.get("oneOf") or []:
        accepted.update((branch or {}).get("required") or [])
    accepted.update(request.get("required") or [])
    accepted.update(request.get("optional") or [])
    return "context_id" in accepted


async def read_partial_trajectory(
    *,
    a2a_url: str,
    a2a_card: dict,
    context_id: str,
    timeout_seconds: float,
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
    params = traj_ext.get("params") or {}
    # Presence, not truthiness: a card may advertise a method as a bare `{}`,
    # which is falsy. Testing truth would read such a card as not having it.
    get_method = (params.get("methods") or {}).get("get")
    if get_method is None:
        return TrajectoryCapture(reason="trajectory_get_unadvertised")
    if not _accepts_context_id(get_method):
        return TrajectoryCapture(reason="trajectory_context_unsupported")

    endpoint = a2a_url + params.get("endpoint", DEFAULT_TRAJECTORY_ENDPOINT)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                endpoint, json={"context_id": context_id}, timeout=timeout_seconds
            )
    except Exception as exc:
        logger.warning("partial trajectory read failed for %s: %s", context_id, exc)
        return TrajectoryCapture(reason="trajectory_read_failed")

    if resp.status_code == 404:
        # No transcript yet — the agent hasn't written a turn.
        return TrajectoryCapture(reason="trajectory_session_missing")
    if resp.status_code >= 400:
        return TrajectoryCapture(reason=f"trajectory_http_{resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        return TrajectoryCapture(reason="trajectory_read_failed")

    trajectory = data.get("trajectory")
    if not trajectory:
        return TrajectoryCapture(reason="trajectory_empty")
    return TrajectoryCapture(trajectory=trajectory)


def upload_trajectory(
    trajectory: Any, trajectory_output_prefix: str, *, name: Optional[str] = None
) -> str:
    """Serialize a trajectory payload and store it, returning its object URL.

    Always uploaded from the worker side, never by the agent: sandbox VMs hold
    static STS env vars with a fixed expiry, and long runs exceed it.

    ``name`` makes the object addressable by an id the caller already records
    (#731); omit it to get a random one, which is what writing repeatedly under a
    single prefix needs, since a fixed name would overwrite the last upload.
    """
    from agent_env.config import get_config

    store = get_config().get_object_store()
    # Via the store, not urlparse().path: a prefix naming a different bucket
    # must fail loudly, not silently write to the configured one.
    prefix_key = store.get_object_key(trajectory_output_prefix)
    if prefix_key and not prefix_key.endswith("/"):
        prefix_key += "/"
    key = f"{prefix_key}trajectory-{name or uuid.uuid4().hex[:12]}.json"
    body = json.dumps(trajectory, indent=2, default=str).encode()
    return store.put(key, body, content_type="application/json")


