"""A run's trigger evidence, the feed the run viewer's Triggers tab polls."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException

from agent_env.config import get_config
from agent_env.env.store import ENV_INSTANCES_COLLECTION
from agent_env.explorer.entity_ids import EntityId
from agent_env.explorer.routers.common import docs
from agent_env.store import Filter, In
from agent_env.task.store import TASK_INSTANCES_COLLECTION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["runs"])

TRIGGER_STATE_TIMEOUT_SECONDS = 2.0
# Both fan-outs are driven by the run context, which a run request can supply.
MAX_TRIGGER_ENVS = 20
TRIGGER_METADATA_KEYS = (
    "env_trigger_registrations",
    "agent_trigger_registrations",
    "agent_trigger_firings",
    "agent_trigger_state",
    "env_trigger_snapshots",
    "usersim_turn_outputs",
    "env_trigger_state",
    "env_trajectory",
)
# Envelope keys the metadata ledger has no copy of.
_ENVELOPE_META_KEYS = ("clock", "clock_read_at_utc")


@router.get("/{task_id}/instances/{instance_id}/triggers")
async def instance_triggers(task_id: EntityId, instance_id: EntityId) -> dict:
    """The run's trigger metadata, plus each env's verbatim ``/triggers/state`` under ``state``:
    from the live gateway while the run is running, else from the envelope ``prompt_agent`` saved.
    An unreachable source degrades to the metadata; ``state_meta`` carries the gateway clock the
    events' ``virtual_time`` is measured on."""
    doc = await asyncio.to_thread(
        docs().find_one, TASK_INSTANCES_COLLECTION, Filter.of(task_id=task_id, instance_id=instance_id)
    )
    if doc is None:
        raise HTTPException(status_code=404, detail=f"instance {instance_id} not found for task {task_id}")

    context = doc.get("context") or {}
    metadata = context.get("metadata") or {}
    ledger = {key: metadata.get(key) for key in TRIGGER_METADATA_KEYS}
    captures = ledger["env_trigger_state"] if isinstance(ledger["env_trigger_state"], dict) else {}
    ledger["env_trigger_state"] = captures
    payload = {
        "status": doc.get("status"),
        "source": "metadata",
        **ledger,
        "state": {},
        "state_sources": {},
        "state_meta": {},
    }

    # Per env: one env's live gateway must not suppress another's envelope.
    if payload["status"] == "running":
        live, live_meta = await _live_trigger_state(context.get("deployed_envs") or [])
        for env_id, state in live.items():
            payload["state_sources"][env_id] = "live" if state is not None else "unavailable"
            if state is not None:
                payload["state"][env_id] = state
        payload["state_meta"].update(live_meta)

    pending = {
        env_id: entry
        for env_id, entry in list(captures.items())[:MAX_TRIGGER_ENVS]
        if isinstance(entry, dict) and env_id not in payload["state"]
    }
    if pending:
        saved, saved_meta = await asyncio.to_thread(_saved_trigger_state, pending, instance_id)
        payload["state"].update(saved)
        payload["state_meta"].update(saved_meta)
        payload["state_sources"].update({env_id: "artifact" for env_id in saved})

    sources = set(payload["state_sources"].values())
    if sources & {"live", "artifact"}:
        payload["source"] = "live" if "live" in sources else "artifact"
    return payload


def _live_gateways(deployed_envs: list) -> list[tuple[str, str | None]]:
    """``(env_id, gateway_url or None)`` per deployed env, from the env-instance records the deploy
    writes rather than the run context's own copy, which a run request can supply."""
    pairs = [
        (env["env_id"], env["instance_id"])
        for env in deployed_envs
        if isinstance(env, dict) and isinstance(env.get("env_id"), str) and isinstance(env.get("instance_id"), str)
    ][:MAX_TRIGGER_ENVS]
    if not pairs:
        return []
    records = docs().query(ENV_INSTANCES_COLLECTION, Filter().where("instance_id", In([iid for _, iid in pairs])))
    urls = {record["instance_id"]: (record.get("gateway_url") or "").rstrip("/") or None for record in records}
    return [(env_id, urls.get(instance_id)) for env_id, instance_id in pairs]


async def _live_trigger_state(deployed_envs: list) -> tuple[dict, dict]:
    """Each env's ``/triggers/state`` and gateway clock, read concurrently. An env with no reachable
    gateway maps to None, so "unreachable" never reads as "no triggers"."""
    targets = await asyncio.to_thread(_live_gateways, deployed_envs)
    state: dict = {env_id: None for env_id, _ in targets}
    meta: dict = {}
    reachable = [(env_id, url) for env_id, url in targets if url]
    if not reachable:
        return state, meta

    async with httpx.AsyncClient(timeout=TRIGGER_STATE_TIMEOUT_SECONDS) as client:

        async def _get(url: str, env_id: str, what: str):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except Exception:
                logger.warning("live %s unavailable for %s", what, env_id, exc_info=True)
                return None

        async def _one(env_id: str, gateway_url: str):
            triggers, clock = await asyncio.gather(
                _get(f"{gateway_url}/triggers/state", env_id, "trigger state"),
                _get(f"{gateway_url}/clock/state", env_id, "clock state"),
            )
            # Stamped at the read: at a high clock rate a second of drift is a virtual day.
            return env_id, triggers, {"clock": clock, "clock_read_at_utc": datetime.now(timezone.utc).isoformat()}

        for env_id, triggers, env_meta in await asyncio.gather(*(_one(e, u) for e, u in reachable)):
            state[env_id] = triggers
            if triggers is not None:
                meta[env_id] = env_meta
    return state, meta


def _saved_trigger_state(entries: dict, instance_id: str) -> tuple[dict, dict]:
    """Each env's ``/triggers/state`` from the envelope ``prompt_agent`` saved, with its capture-time
    clock. Skips a url outside the configured object store, an unreadable envelope, and one another
    instance captured (a resumed run's context carries the earlier attempt's pointers)."""
    state: dict = {}
    meta: dict = {}
    for env_id, entry in entries.items():
        object_url = entry.get("object_url")
        if not isinstance(object_url, str):
            continue
        store = get_config().get_object_store_at(object_url)
        if not store.owns(object_url):
            logger.warning("refusing a trigger-state url outside the object store for %s", env_id)
            continue
        try:
            envelope = json.loads(store.get(object_url))
            captured_by = envelope.get("instance_id")
            if captured_by and captured_by != instance_id:
                logger.warning("skipping the trigger state for %s captured by %s", env_id, captured_by)
                continue
            state[env_id] = envelope.get("state") or {}
            meta[env_id] = {key: envelope.get(key) for key in _ENVELOPE_META_KEYS}
        except Exception:
            logger.warning("trigger-state envelope unreadable for %s", env_id, exc_info=True)
    return state, meta
