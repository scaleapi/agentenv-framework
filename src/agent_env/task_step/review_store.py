"""Store for human-in-the-loop review decisions, keyed by ``(instance_id, step_id)``.

Both the worker (``ReviewTaskStep``) and the hub write here, so it lives in
agent-env and the hub imports it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from agent_env.store import DuplicateKeyError, Eq, Filter, UpdateSpec, get_config

logger = logging.getLogger(__name__)

REVIEWS_COLLECTION = "task_reviews"

AWAITING = "awaiting"
CONTINUE = "continue"
ABORT = "abort"
_DECISIONS = (CONTINUE, ABORT)

# Reap old coordination docs; window dwarfs any realistic review timeout, so active reviews never expire.
_TTL = timedelta(days=7)


class ReviewStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(
                REVIEWS_COLLECTION, ["instance_id", "step_id"], unique=True
            )
            store.ensure_index(
                REVIEWS_COLLECTION, ["expires_at"], ttl_seconds=0
            )
            self._indexed = store
        return store

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def put_awaiting(
        self,
        instance_id: str,
        step_id: str,
        *,
        label: Optional[str] = None,
        upstream: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a review as awaiting a decision. Idempotent — a retry won't clobber a recorded decision."""
        doc = {
            "instance_id": instance_id,
            "step_id": step_id,
            "state": AWAITING,
            "label": label,
            "upstream": upstream or {},
            "awaiting_at": self._now(),
            "expires_at": datetime.now(timezone.utc) + _TTL,
        }
        try:
            self._doc_store.insert(REVIEWS_COLLECTION, doc)
        except DuplicateKeyError:
            pass

    def get(self, instance_id: str, step_id: str) -> Optional[dict[str, Any]]:
        return self._doc_store.find_one(
            REVIEWS_COLLECTION, Filter.of(instance_id=instance_id, step_id=step_id)
        )

    def set_decision(
        self,
        instance_id: str,
        step_id: str,
        decision: str,
        *,
        decided_by: Optional[str] = None,
    ) -> bool:
        """Record ``continue``/``abort`` on an awaiting review; False if none was awaiting."""
        if decision not in _DECISIONS:
            raise ValueError(f"decision must be one of {_DECISIONS}, got {decision!r}")
        matched = self._doc_store.update(
            REVIEWS_COLLECTION,
            Filter.of(instance_id=instance_id, step_id=step_id).where("state", Eq(AWAITING)),
            UpdateSpec(
                set={"state": decision, "decided_by": decided_by, "decided_at": self._now()}
            ),
        )
        return matched > 0

    def list_awaiting(self, instance_id: Optional[str] = None) -> list[dict[str, Any]]:
        filt = Filter.of(state=AWAITING)
        if instance_id is not None:
            filt = filt.where("instance_id", Eq(instance_id))
        return self._doc_store.query(REVIEWS_COLLECTION, filt)


_store: Optional[ReviewStore] = None


def get_review_store() -> ReviewStore:
    global _store
    if _store is None:
        _store = ReviewStore()
    return _store


def reset_review_store() -> None:
    global _store
    _store = None
