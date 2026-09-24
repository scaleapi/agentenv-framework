"""DocumentStore-backed store for A2A conversations.

Each conversation is a multi-turn A2A interaction keyed by `conversation_id`
(== A2A `contextId`). v1 use case is human-in-the-loop hosted by the
hub backend; future services that host an A2A endpoint can import
this module and operate on the same collection.

A conversation contains an append-only `messages` array plus a parallel
`a2a_tasks` array tracking each A2A round-trip (one a2a_task per
`message/send`). The `a2a_` prefix disambiguates these protocol-level tasks
from agent-env's own `Task` / `task_instance_id` vocabulary. Each a2a_task
carries `input_message_idx` / `response_message_idx` pointers into
`messages` so a `tasks/get` poll can resolve to the right response without
scanning.

All operations are synchronous. Call from `async def` handlers via
`loop.run_in_executor(...)` to avoid blocking the event loop.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from a2a.types import TaskState

from agent_env.store import (
    Eq,
    Filter,
    Sort,
    UpdateSpec,
    compare_and_swap,
    get_config,
    rev_precondition,
)

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "agent_env_a2a_conversations"
_REV = "rev"

_MAX_CAS_ATTEMPTS = 100


def _doc_store():
    return get_config().get_document_store()


def ensure_indexes() -> None:
    """Create indexes for the a2a_conversations collection. Idempotent.

    No TTL — conversations are retained indefinitely for audit and replay.
    """
    docs = _doc_store()
    docs.ensure_index(_COLLECTION_NAME, ["conversation_id"], unique=True)
    docs.ensure_index(_COLLECTION_NAME, ["task_instance_id"])
    docs.ensure_index(_COLLECTION_NAME, ["a2a_tasks.a2a_task_id"])
    docs.ensure_index(_COLLECTION_NAME, ["pending", "status", "updated_at_utc"])


def create_conversation(
    conversation_id: str,
    task_instance_id: str,
    source_agent_name: str,
    target_agent_name: str,
) -> dict:
    """Insert a new conversation in the `active` state with empty messages and a2a_tasks.

    `source_agent_name` identifies the initiator (the side that calls first).
    `target_agent_name` identifies the responder (the side being called). For
    the canonical PromptAgent multi-turn case: source="human_agent" (or a
    user-sim name like "hana_kim"), target=the solver's in-task DAG name. For
    HITL where a solver consults a human peer: source=the solver's name,
    target="human_agent".

    Raises `agent_env.store.DuplicateKeyError` if `conversation_id` already
    exists — callers should treat that as "use the existing conversation."
    """
    now = datetime.now(timezone.utc)
    doc = {
        "conversation_id": conversation_id,
        "task_instance_id": task_instance_id,
        "source_agent_name": source_agent_name,
        "target_agent_name": target_agent_name,
        "messages": [],
        "a2a_tasks": [],
        "pending": False,
        "status": "active",
        _REV: 0,
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    _doc_store().insert(_COLLECTION_NAME, doc)
    return doc


def get_conversation(conversation_id: str) -> Optional[dict]:
    """Return the conversation by id, or None if not found."""
    return _doc_store().find_one(_COLLECTION_NAME, Filter.of(conversation_id=conversation_id))


def find_by_a2a_task_id(a2a_task_id: str) -> Optional[dict]:
    """Lookup the conversation containing an a2a_task with the given id.

    Used by `tasks/get` polls — the multikey index on `a2a_tasks.a2a_task_id`
    makes this a single-key index hit even though `a2a_tasks` is an array.
    """
    return _doc_store().find_one(
        _COLLECTION_NAME, Filter().where("a2a_tasks.a2a_task_id", Eq(a2a_task_id))
    )


def add_a2a_task(
    conversation_id: str,
    parts: list[dict],
    a2a_task_id: str,
    role: str,
) -> Optional[dict]:
    """Append a caller-side message and create a new working a2a_task.

    Called whenever an A2A `message/send` fires. The caller's role depends on
    the conversation direction:
      - PromptAgent multi-turn (canonical): user is the caller → role="user"
      - HITL (solver consulting human peer): solver is the caller → role="agent"

    Reads the conversation to compute `input_message_idx` (the index the new
    message lands at), then appends both the message and the new a2a_task under
    a `rev` compare-and-swap. A concurrent write that landed first bumps `rev`,
    fails the swap, and we retry with a fresh index, so `input_message_idx`
    always names the message we actually appended. Both arrays are pushed
    (never rewritten), so no entry is dropped.

    Returns the post-update conversation doc, or None if `conversation_id`
    doesn't exist.
    """
    now = datetime.now(timezone.utc)
    new_message = {"role": role, "parts": parts, "ts": now.isoformat()}

    def _mutate(doc: dict) -> UpdateSpec:
        idx = len(doc.get("messages", []))
        new_a2a_task = {
            "a2a_task_id": a2a_task_id,
            "state": TaskState.working.value,
            "requested_at": now.isoformat(),
            "input_message_idx": idx,
        }
        return UpdateSpec(
            push={"messages": [new_message], "a2a_tasks": [new_a2a_task]},
            set={"pending": True, "updated_at_utc": now},
        )

    return compare_and_swap(
        _doc_store(),
        _COLLECTION_NAME,
        Filter.of(conversation_id=conversation_id),
        _mutate,
        counter_field=_REV,
    )


def complete_a2a_task(
    conversation_id: str,
    parts: list[dict],
    role: str,
) -> Optional[dict]:
    """Append a responder-side message and complete the oldest working a2a_task.

    Called when the responder replies. The responder's role depends on the
    conversation direction:
      - PromptAgent multi-turn (canonical): agent is the responder → role="agent"
      - HITL (human replying to solver's consult): human is the responder → role="user"

    Compare-and-swap on the conversation `rev`, so `response_message_idx` names
    the message we actually appended even if a concurrent write raced us (we
    retry with a fresh index). The target a2a_task is additionally guarded on
    `state == "working"`. A failed swap is disambiguated by a re-read: a
    concurrent write that left the target working → retry; the target no longer
    working → return None (the message is not appended, matching the pre-port
    guarded update).

    If there is no working a2a_task (the responder is replying out-of-band with
    no in-flight A2A request), the message is appended and no a2a_task is
    touched.

    Returns the post-update conversation doc, or None if `conversation_id`
    doesn't exist (or the target task lost the state race).
    """
    now = datetime.now(timezone.utc)
    new_message = {"role": role, "parts": parts, "ts": now.isoformat()}
    # One backend for the whole retry loop: a reset between the read and the CAS would
    # decide against one database and write to another.
    docs = _doc_store()

    for _ in range(_MAX_CAS_ATTEMPTS):
        existing = docs.find_one(
            _COLLECTION_NAME, Filter.of(conversation_id=conversation_id)
        )
        if existing is None:
            return None

        idx = len(existing.get("messages", []))
        oldest_working_idx: Optional[int] = next(
            (
                i
                for i, t in enumerate(existing.get("a2a_tasks", []))
                if t.get("state") == TaskState.working.value
            ),
            None,
        )

        spec = UpdateSpec(
            push={"messages": [new_message]},
            set={"pending": False, "updated_at_utc": now},
            inc={_REV: 1},
        )
        filter_ = Filter.of(conversation_id=conversation_id).where(
            _REV, rev_precondition(existing, _REV)
        )
        if oldest_working_idx is not None:
            spec.set[f"a2a_tasks.{oldest_working_idx}.state"] = TaskState.completed.value
            spec.set[f"a2a_tasks.{oldest_working_idx}.response_message_idx"] = idx
            spec.set[f"a2a_tasks.{oldest_working_idx}.ended_at"] = now.isoformat()
            filter_ = filter_.where(
                f"a2a_tasks.{oldest_working_idx}.state", Eq(TaskState.working.value)
            )

        result = docs.update_one_and_get(_COLLECTION_NAME, filter_, spec)
        if result is not None:
            return result

        if oldest_working_idx is None:
            continue

        recheck = docs.find_one(
            _COLLECTION_NAME, Filter.of(conversation_id=conversation_id)
        )
        if recheck is None:
            return None
        tasks = recheck.get("a2a_tasks", [])
        still_working = (
            oldest_working_idx < len(tasks)
            and tasks[oldest_working_idx].get("state") == TaskState.working.value
        )
        if not still_working:
            return None
    raise RuntimeError(
        f"complete_a2a_task: rev CAS did not converge for conversation_id={conversation_id}"
    )


def cancel_a2a_task(
    conversation_id: str,
    a2a_task_id: str,
) -> Optional[dict]:
    """Mark a specific a2a_task as canceled. Used by A2A `tasks/cancel`.

    Does not append a message; only updates the a2a_task entry. Locates the
    target by id in Python, then writes under an `a2a_tasks.<i>.state ==
    "working"` precondition so a task that already terminated (or a racing
    writer) yields a no-op.

    Returns the post-update conversation doc, or None if the conversation or
    task isn't found / wasn't in `working` state.
    """
    now = datetime.now(timezone.utc)
    docs = _doc_store()
    existing = docs.find_one(_COLLECTION_NAME, Filter.of(conversation_id=conversation_id))
    if existing is None:
        return None

    target_idx: Optional[int] = next(
        (
            i
            for i, t in enumerate(existing.get("a2a_tasks", []))
            if t.get("a2a_task_id") == a2a_task_id and t.get("state") == TaskState.working.value
        ),
        None,
    )
    if target_idx is None:
        return None

    return docs.update_one_and_get(
        _COLLECTION_NAME,
        Filter.of(conversation_id=conversation_id).where(
            f"a2a_tasks.{target_idx}.state", Eq(TaskState.working.value)
        ),
        UpdateSpec(
            set={
                f"a2a_tasks.{target_idx}.state": TaskState.canceled.value,
                f"a2a_tasks.{target_idx}.ended_at": now.isoformat(),
                "updated_at_utc": now,
            },
            inc={_REV: 1},
        ),
    )


def mark_closed(conversation_id: str) -> Optional[dict]:
    """Set `status=closed` and cancel any working a2a_tasks.

    Reads the conversation, cancels working a2a_tasks in Python, and writes the
    transformed array back under a `rev` compare-and-swap so a concurrently
    appended a2a_task (e.g. a `start_a2a_task` push that lands between the read
    and the write) is not clobbered by this whole-array write. On a lost swap we
    re-read and re-transform; a task that landed mid-close is then canceled too,
    which is correct for a terminal close.

    Returns the post-update conversation doc, or None if not found.
    """
    docs = _doc_store()
    for _ in range(_MAX_CAS_ATTEMPTS):
        now = datetime.now(timezone.utc)
        existing = docs.find_one(
            _COLLECTION_NAME, Filter.of(conversation_id=conversation_id)
        )
        if existing is None:
            return None

        new_tasks = [
            {**t, "state": TaskState.canceled.value, "ended_at": now.isoformat()}
            if t.get("state") == TaskState.working.value
            else t
            for t in existing.get("a2a_tasks", [])
        ]
        result = docs.update_one_and_get(
            _COLLECTION_NAME,
            Filter.of(conversation_id=conversation_id).where(
                _REV, rev_precondition(existing, _REV)
            ),
            UpdateSpec(
                set={
                    "status": "closed",
                    "pending": False,
                    "updated_at_utc": now,
                    "a2a_tasks": new_tasks,
                },
                inc={_REV: 1},
            ),
        )
        if result is not None:
            return result
    raise RuntimeError(
        f"mark_closed: rev CAS did not converge for conversation_id={conversation_id}"
    )


def list_pending(limit: int = 50) -> list[dict]:
    """Return active conversations awaiting a user reply, most-recently-updated first.

    Backs the hub-frontend Conversations worklist.
    """
    return _doc_store().query(
        _COLLECTION_NAME,
        Filter.of(pending=True, status="active"),
        sort=Sort.by("updated_at_utc", descending=True),
        limit=limit,
    )
