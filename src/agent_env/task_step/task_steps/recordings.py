"""What a run's recordings share, a computer's (``computer_recording``) and a website browser's
(``browser.recording``): where each uploads and the grant it uploads under, the error its entry keeps, and teardown's
two passes over the ones a run left open, the first ending them, the second registering from its namespace any still
open once the envs have closed."""
from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from agent_env.a2a_agent.object_transfer import ObjectLimits, namespace_grant
from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.store.base import GrantUnavailableError
from agent_env.store.object_store.object_store import issues_grants_to
from agent_env.task_step.context import TaskStepContext

logger = logging.getLogger(__name__)

#: How much longer than the env's ``max_duration_s`` a recording's grant lasts.
GRANT_MARGIN_S = 600
#: How much of an error a recording's entry keeps.
ERROR_CHARS = 500


def why(error: BaseException) -> str:
    """``error`` as a recording's entry keeps it: its type and message, cut to ``ERROR_CHARS``."""
    return f"{type(error).__name__}: {error}"[:ERROR_CHARS]


def refuse_no_network(deployed) -> None:
    """A computer deployed with no network can't reach the store its recordings upload to."""
    network = ((deployed.metadata or {}).get("computer") or {}).get("network")
    if network and NetworkPolicy.from_dict(network).closed:
        raise RuntimeError(f"env {deployed.env_id!r} was deployed with no network, so its computer can't upload a "
                           "recording")


def recording_namespace(request_id: str, prefix: str) -> str:
    """Where a recording's files go, under ``prefix``: named by its start's request id, so a retried start uploads
    alongside."""
    config = get_config()
    key = uuid.uuid5(uuid.NAMESPACE_URL, request_id).hex
    return config.get_object_store().object_url(f"{config.get_artifact_key_prefix()}{prefix}/{key}")


def recording_grant(deployed, params: dict, namespace: str, max_objects: int, max_object_bytes: int,
                    max_total_bytes: int) -> dict:
    """A grant that uploads under ``namespace``, within the limits given and the store's own, lasting the recording
    extension's ``max_duration_s`` (``params``) and a margin, or the longest the store issues; refused when the store
    issues no grant that reaches the env's sandbox."""
    store = get_config().get_object_store()
    sandbox_type = getattr(deployed, "sandbox_type", None)
    if not issues_grants_to(store, sandbox_type):
        raise RuntimeError(f"the object store issues no upload grant that reaches env {deployed.env_id}'s sandbox "
                           f"({sandbox_type or 'unknown'}), so its recording couldn't upload")
    limits = ObjectLimits(max_objects=max_objects, max_object_bytes=min(store.max_single_upload_bytes or max_object_bytes,
                                                                        max_object_bytes),
                          max_total_bytes=max_total_bytes)
    try:
        grant = namespace_grant(store, namespace, limits=limits, expires_in=params["max_duration_s"] + GRANT_MARGIN_S)
    except GrantUnavailableError:
        grant = namespace_grant(store, namespace, limits=limits, expires_in=store.grant_lifetime_seconds)
    return grant.model_dump(mode="json")


def open_entries(context: TaskStepContext, key: str, env_ids: Optional[list[str]]) -> list[tuple[str, dict]]:
    """The recordings ``context.metadata[key]`` holds that are still open, by name, of ``env_ids`` only when given."""
    return [(name, entry) for name, entry in (context.metadata.get(key) or {}).items()
            if entry.get("state") == "open" and (env_ids is None or entry["env_id"] in env_ids)]


async def end_open(context: TaskStepContext, key: str, env_ids: Optional[list[str]],
                   end: Callable[[TaskStepContext, str, str], Awaitable[None]]) -> None:
    """``end`` each recording still open, for teardown."""
    for name, _ in open_entries(context, key, env_ids):
        await end(context, name, "teardown")


def register_open(context: TaskStepContext, key: str, env_ids: Optional[list[str]],
                  from_namespace: Callable[[dict], Any], registered: Callable[[TaskStepContext, str, dict, Any], None],
                  label: str) -> None:
    """Register each recording still open from what its namespace holds, once its env has closed: ``from_namespace``
    reads its manifest there and ``registered`` registers it. Never raises: each entry says what happened."""
    for name, entry in open_entries(context, key, env_ids):
        try:
            manifest = from_namespace(entry)
        except Exception as e:
            logger.warning("%s %s: listing its namespace failed: %s", label, name, e)
            entry["state"], entry["error"] = "failed", why(e)
            continue
        registered(context, name, entry, manifest)
