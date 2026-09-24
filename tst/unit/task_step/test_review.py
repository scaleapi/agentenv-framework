"""Unit tests for ReviewTaskStep — poll -> continue / abort / timeout."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_env.task_step.review_store import ABORT, AWAITING, CONTINUE, ReviewStore
from agent_env.config import set_document_store
from agent_env.task_step.task_steps.review import (
    ReviewAbortedError,
    ReviewTaskStep,
)


class FakeStore:
    """Returns successive states from `states` (repeats the last one)."""

    def __init__(self, states):
        self._states = list(states)
        self.awaiting_calls = []

    def put_awaiting(self, instance_id, step_id, *, label=None, upstream=None):
        self.awaiting_calls.append(
            {"instance_id": instance_id, "step_id": step_id, "label": label, "upstream": upstream}
        )

    def get(self, instance_id, step_id):
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return {"state": state, "decided_by": "op@example.com"}


def _ctx(prompt_responses=None):
    return SimpleNamespace(
        instance_id="inst-1",
        metadata={"_heartbeat_fn": lambda: None},
        prompt_responses=prompt_responses or [],
    )


def _step(**kwargs):
    kwargs.setdefault("poll_interval_seconds", 0.01)
    kwargs.setdefault("timeout_seconds", 5)
    return ReviewTaskStep(id="cp1", version=None, **kwargs)


@pytest.mark.asyncio
async def test_continue_returns_context_and_records_decision():
    store = FakeStore([AWAITING, CONTINUE])
    ctx = _ctx()
    with patch(
        "agent_env.task_step.review_store.get_review_store", return_value=store
    ):
        result = await _step().execute(ctx)

    assert result is ctx
    assert ctx.metadata["review_decisions"]["cp1"] == {
        "decision": CONTINUE,
        "decided_by": "op@example.com",
    }
    assert store.awaiting_calls[0]["instance_id"] == "inst-1"
    assert store.awaiting_calls[0]["step_id"] == "cp1"


@pytest.mark.asyncio
async def test_abort_raises_and_never_returns_context():
    store = FakeStore([ABORT])
    with patch(
        "agent_env.task_step.review_store.get_review_store", return_value=store
    ):
        with pytest.raises(ReviewAbortedError):
            await _step().execute(_ctx())


@pytest.mark.asyncio
async def test_timeout_raises():
    store = FakeStore([AWAITING])
    with patch(
        "agent_env.task_step.review_store.get_review_store", return_value=store
    ):
        with pytest.raises(TimeoutError):
            await _step(timeout_seconds=0).execute(_ctx())


@pytest.mark.asyncio
async def test_zero_poll_interval_still_times_out():
    # A non-positive poll interval must still hit the wall-clock deadline, not spin forever.
    store = FakeStore([AWAITING])
    with patch(
        "agent_env.task_step.review_store.get_review_store", return_value=store
    ):
        with pytest.raises(TimeoutError):
            await _step(poll_interval_seconds=0, timeout_seconds=0.05).execute(_ctx())


@pytest.mark.asyncio
async def test_missing_instance_id_raises():
    ctx = _ctx()
    ctx.instance_id = None
    with pytest.raises(RuntimeError, match="instance_id"):
        await _step().execute(ctx)


@pytest.mark.asyncio
async def test_upstream_summary_surfaces_structured_verdict():
    responses = [
        SimpleNamespace(step_id="m1", structured_output={"verdict": "REJECT", "quality_score": 0.4}, response=""),
    ]
    store = FakeStore([CONTINUE])
    with patch(
        "agent_env.task_step.review_store.get_review_store", return_value=store
    ):
        await _step().execute(_ctx(prompt_responses=responses))

    upstream = store.awaiting_calls[0]["upstream"]
    assert upstream["step_id"] == "m1"
    assert upstream["structured_output"]["verdict"] == "REJECT"


@pytest.mark.asyncio
async def test_upstream_summary_bounds_oversized_structured_output():
    huge = {"items": ["x" * 1000 for _ in range(100)]}
    responses = [SimpleNamespace(step_id="m1", structured_output=huge, response="")]
    store = FakeStore([CONTINUE])
    with patch(
        "agent_env.task_step.review_store.get_review_store", return_value=store
    ):
        await _step().execute(_ctx(prompt_responses=responses))

    surfaced = store.awaiting_calls[0]["upstream"]["structured_output"]
    assert surfaced["_truncated"] is True
    assert len(surfaced["preview"]) == 16000


def test_put_awaiting_stamps_future_datetime_expiry_for_ttl():
    # expires_at must be a real datetime, not an isoformat string, or the Mongo TTL index silently never reaps.
    captured: dict = {}

    class FakeDocStore:
        def insert(self, collection, doc):
            captured.update(doc)

        def ensure_index(self, *a, **k):   # a DocumentStore has one; this double stands in for it
            pass

    store = ReviewStore()
    set_document_store(FakeDocStore())
    store.put_awaiting("inst-1", "cp1", label="x", upstream={})

    expires_at = captured["expires_at"]
    assert isinstance(expires_at, datetime)
    assert expires_at > datetime.now(timezone.utc)
