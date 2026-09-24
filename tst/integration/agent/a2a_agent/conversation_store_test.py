"""Integration coverage for the DocumentStore-backed A2A conversation store.

Runs against the configured document store. Exercises the read→compute-in-Python→guarded-write
rewrite of the former aggregation-pipeline updates, including the concurrency
property the pipeline used to give for free (parallel writers must not drop
appended entries, and racing terminal transitions must converge to exactly one
terminal state).
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import conversation_store as cs
from agent_env.store import DuplicateKeyError, Filter, get_config


def _docs():
    return get_config().get_document_store()


@pytest.fixture(scope="module", autouse=True)
def _indexes():
    cs.ensure_indexes()


@pytest.fixture
def conversation_id():
    cid = f"conv-test-{uuid.uuid4().hex[:10]}"
    yield cid
    _docs().delete(cs._COLLECTION_NAME, Filter.of(conversation_id=cid))


def _new(cid: str) -> dict:
    return cs.create_conversation(
        conversation_id=cid,
        task_instance_id="ti-test",
        source_agent_name="human_agent",
        target_agent_name="solver",
    )


@pytest.mark.integration
def test_create_and_get_roundtrip(conversation_id):
    created = _new(conversation_id)
    assert created["status"] == "active"
    assert created["messages"] == [] and created["a2a_tasks"] == []
    assert "_id" not in created

    fetched = cs.get_conversation(conversation_id)
    assert fetched is not None and fetched["conversation_id"] == conversation_id
    assert cs.get_conversation("nonexistent") is None


@pytest.mark.integration
def test_duplicate_create_raises_abstraction_duplicate_key(conversation_id):
    _new(conversation_id)
    with pytest.raises(DuplicateKeyError):
        _new(conversation_id)


@pytest.mark.integration
def test_add_then_complete_pins_message_indices(conversation_id):
    _new(conversation_id)
    task_id = uuid.uuid4().hex

    after_add = cs.add_a2a_task(conversation_id, [{"kind": "text", "text": "hi"}], task_id, "user")
    assert after_add["pending"] is True
    assert len(after_add["messages"]) == 1
    task = after_add["a2a_tasks"][0]
    assert task["state"] == TaskState.working.value
    assert task["input_message_idx"] == 0

    after_complete = cs.complete_a2a_task(conversation_id, [{"kind": "text", "text": "yo"}], "agent")
    assert after_complete["pending"] is False
    assert len(after_complete["messages"]) == 2
    task = after_complete["a2a_tasks"][0]
    assert task["state"] == TaskState.completed.value
    assert task["response_message_idx"] == 1
    assert cs.find_by_a2a_task_id(task_id)["conversation_id"] == conversation_id


@pytest.mark.integration
def test_add_a2a_task_migrates_pre_rev_conversation(conversation_id):
    """A conversation written before `rev` existed still accepts writes: the CAS
    matches an absent rev, and the inc migrates the doc to rev=1."""
    _docs().insert(cs._COLLECTION_NAME, {
        "conversation_id": conversation_id,
        "task_instance_id": "ti",
        "source_agent_name": "human_agent",
        "target_agent_name": "solver",
        "messages": [],
        "a2a_tasks": [],
        "pending": False,
        "status": "active",
    })  # note: no "rev" field, like a pre-port doc

    after = cs.add_a2a_task(conversation_id, [{"kind": "text", "text": "hi"}], uuid.uuid4().hex, "user")
    assert after is not None
    assert after["a2a_tasks"][0]["input_message_idx"] == 0
    assert after["rev"] == 1


@pytest.mark.integration
def test_cancel_marks_task_and_appends_no_message(conversation_id):
    _new(conversation_id)
    task_id = uuid.uuid4().hex
    cs.add_a2a_task(conversation_id, [{"kind": "text", "text": "hi"}], task_id, "user")

    after = cs.cancel_a2a_task(conversation_id, task_id)
    assert after["a2a_tasks"][0]["state"] == TaskState.canceled.value
    assert len(after["messages"]) == 1  # cancel appends nothing
    # Cancelling an already-terminal task is a no-op.
    assert cs.cancel_a2a_task(conversation_id, task_id) is None


@pytest.mark.integration
def test_mark_closed_cancels_working_tasks(conversation_id):
    _new(conversation_id)
    cs.add_a2a_task(conversation_id, [{"kind": "text", "text": "hi"}], uuid.uuid4().hex, "user")

    after = cs.mark_closed(conversation_id)
    assert after["status"] == "closed"
    assert after["pending"] is False
    assert after["a2a_tasks"][0]["state"] == TaskState.canceled.value
    # Idempotent: closing again still returns the doc.
    assert cs.mark_closed(conversation_id)["status"] == "closed"


@pytest.mark.integration
def test_list_pending_surfaces_pending_active(conversation_id):
    _new(conversation_id)
    cs.add_a2a_task(conversation_id, [{"kind": "text", "text": "hi"}], uuid.uuid4().hex, "user")
    pending_ids = {c["conversation_id"] for c in cs.list_pending(limit=200)}
    assert conversation_id in pending_ids


@pytest.mark.integration
def test_concurrent_add_a2a_task_keeps_message_indices_correct(conversation_id):
    """N parallel message/send calls must append all N tasks + messages AND give
    each a2a_task a correct, distinct `input_message_idx`.

    The old $concatArrays pipeline computed the index atomically server-side;
    the port restores that under concurrency via the `rev` compare-and-swap.
    Without it, racing readers would stamp duplicate indices.
    """
    _new(conversation_id)
    n = 12
    task_ids = [uuid.uuid4().hex for _ in range(n)]

    def add(tid: str):
        return cs.add_a2a_task(conversation_id, [{"kind": "text", "text": tid}], tid, "user")

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(add, task_ids))

    doc = cs.get_conversation(conversation_id)
    assert len(doc["messages"]) == n
    assert len(doc["a2a_tasks"]) == n
    assert {t["a2a_task_id"] for t in doc["a2a_tasks"]} == set(task_ids)

    # Each input_message_idx is distinct and covers exactly 0..n-1, and points
    # at the message that a2a_task actually appended (matched via the text tag).
    indices = sorted(t["input_message_idx"] for t in doc["a2a_tasks"])
    assert indices == list(range(n)), indices
    for t in doc["a2a_tasks"]:
        assert doc["messages"][t["input_message_idx"]]["parts"][0]["text"] == t["a2a_task_id"]


@pytest.mark.integration
def test_concurrent_complete_and_cancel_converge_to_one_terminal_state(conversation_id):
    """A complete racing a cancel on the same working task: exactly one wins,
    the task ends terminal (never left `working`, never double-transitioned)."""
    _new(conversation_id)
    task_id = uuid.uuid4().hex
    cs.add_a2a_task(conversation_id, [{"kind": "text", "text": "hi"}], task_id, "user")

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_complete = ex.submit(cs.complete_a2a_task, conversation_id, [{"kind": "text", "text": "done"}], "agent")
        f_cancel = ex.submit(cs.cancel_a2a_task, conversation_id, task_id)
        f_complete.result()
        f_cancel.result()

    task = cs.get_conversation(conversation_id)["a2a_tasks"][0]
    assert task["state"] in (TaskState.completed.value, TaskState.canceled.value)
    assert task["state"] != TaskState.working.value
