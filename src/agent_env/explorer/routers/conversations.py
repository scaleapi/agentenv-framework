"""Read-only A2A conversation transcripts, per task instance."""

from __future__ import annotations

from fastapi import APIRouter

from agent_env.explorer.entity_ids import EntityId
from agent_env.explorer.routers.common import docs
from agent_env.store import Filter, Sort

# Prefix hardcoded (not imported from app) to avoid a router->app import cycle.
router = APIRouter(prefix="/api/v1", tags=["conversations"])

# Written by agent_env.a2a_agent.conversation_store.
_CONVERSATIONS_COLLECTION = "agent_env_a2a_conversations"


@router.get("/task-instances/{task_instance_id}/conversations")
def list_conversations(task_instance_id: EntityId) -> dict:
    """A2A conversations recorded for a task instance, oldest first."""
    conversations = docs().query(
        _CONVERSATIONS_COLLECTION,
        Filter.of(task_instance_id=task_instance_id),
        sort=Sort.by("created_at_utc", descending=False),  # oldest first
    )
    return {"conversations": conversations}
