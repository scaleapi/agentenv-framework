"""The step that owns a conversation, and closing what an earlier run of that step left active."""

import uuid

from a2a.types import TaskState

from agent_env.a2a_agent import conversation_store as cs


def _new(instance: str, step_id: str | None = None) -> str:
    conversation_id = uuid.uuid4().hex
    cs.create_conversation(conversation_id, instance, "user", "solver", step_id=step_id)
    return conversation_id


def _status(conversation_id: str) -> str:
    return cs.get_conversation(conversation_id)["status"]


def test_a_conversation_records_its_step_only_when_given_one(local_stores):
    with_step, without = _new("ti", step_id="solve"), _new("ti")

    assert cs.get_conversation(with_step)["step_id"] == "solve"
    assert "step_id" not in cs.get_conversation(without)


def test_closing_a_steps_leftovers_closes_only_its_active_conversations(local_stores):
    leftover = _new("ti", step_id="solve")
    cs.add_a2a_task(leftover, [{"kind": "text", "text": "hi"}], "turn", "user")
    already_closed = _new("ti", step_id="solve")
    cs.mark_closed(already_closed)
    other_step, other_instance, unowned = _new("ti", step_id="other"), _new("ti-2", step_id="solve"), _new("ti")

    assert cs.close_active_for_step("ti", "solve") == [leftover]

    closed = cs.get_conversation(leftover)
    assert closed["status"] == "closed"
    assert closed["a2a_tasks"][0]["state"] == TaskState.canceled.value
    assert [_status(c) for c in (other_step, other_instance, unowned)] == ["active"] * 3
    assert cs.close_active_for_step("ti", "solve") == []
