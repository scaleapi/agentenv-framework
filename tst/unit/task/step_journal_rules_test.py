"""The journal's replay rules on their own: plain dicts in, a context out. Anything that needs
a store to be true lives in step_journal_test.py."""

from __future__ import annotations

from agent_env.task.step_journal import (
    _SEED_STEP_ID,
    _set_path,
    _union,
    _unset_path,
    commit_ordered,
    replay_context,
)
from agent_env.task_step.context_ops import ContextUpdateOps


def _entry(step_id: str, seq: int, *, sets=None, unsets=(), adds=None) -> dict:
    ops = ContextUpdateOps(sets=dict(sets or {}), unsets=set(unsets), add_to_sets=dict(adds or {}))
    return {"step_id": step_id, "seq": seq, "ops": ops.to_journal_dict()}


def _completed(*step_ids: str) -> list[dict]:
    return [{"step_id": s, "status": "success"} for s in step_ids]


def _ids(entries: list[dict]) -> list[str]:
    return [e["step_id"] for e in entries]


# ── commit_ordered ───────────────────────────────────────────────────────────

def test_the_seed_comes_first_wherever_it_sits_among_the_entries():
    entries = [_entry("a", 2), _entry(_SEED_STEP_ID, 1), _entry("b", 3)]
    assert _ids(commit_ordered(entries, _completed("a", "b"))) == [_SEED_STEP_ID, "a", "b"]


def test_entries_follow_their_completion_order_not_their_seq():
    # a took the lower seq but b's completion landed first, so b replays first.
    entries = [_entry(_SEED_STEP_ID, 1), _entry("a", 2), _entry("b", 3)]
    assert _ids(commit_ordered(entries, _completed("b", "a"))) == [_SEED_STEP_ID, "b", "a"]


def test_an_entry_whose_step_never_completed_is_dropped():
    entries = [_entry(_SEED_STEP_ID, 1), _entry("a", 2), _entry("uncommitted", 3)]
    assert _ids(commit_ordered(entries, _completed("a"))) == [_SEED_STEP_ID, "a"]


def test_a_re_recorded_step_replays_where_its_latest_completion_sits():
    # The completion moves to the end on a re-record, so the entry must too.
    entries = [_entry(_SEED_STEP_ID, 1), _entry("s0", 4), _entry("s1", 3)]
    assert _ids(commit_ordered(entries, _completed("s1", "s0"))) == [_SEED_STEP_ID, "s1", "s0"]


# ── replay_context ───────────────────────────────────────────────────────────

def test_it_folds_sets_unsets_and_adds_over_an_empty_context():
    entries = [
        _entry(_SEED_STEP_ID, 1, sets={"context.metadata.keep": 1, "context.metadata.drop": 2}),
        _entry("a", 2, sets={"context.metadata.keep": 9}, unsets=["context.metadata.drop"],
               adds={"context.metadata.l": ["x"]}),
    ]
    ctx = replay_context(commit_ordered(entries, _completed("a")), [])
    assert ctx["metadata"] == {"keep": 9, "l": ["x"]}


def test_it_starts_from_a_default_context():
    ctx = replay_context([], [])
    assert ctx["deployed_envs"] == [] and ctx["metadata"] == {} and ctx["agent_model"] is None


def test_an_empty_entry_contributes_nothing():
    # What a completion carried in by start_step journals: the seed already holds its writes.
    entries = [_entry(_SEED_STEP_ID, 1, sets={"context.metadata.base": 1}), _entry("seeded", 2)]
    assert replay_context(commit_ordered(entries, _completed("seeded")), [])["metadata"] == {"base": 1}


def test_the_last_write_of_a_shared_path_wins_in_commit_order():
    entries = [_entry(_SEED_STEP_ID, 1), _entry("a", 2, sets={"context.metadata.p": "A"}),
               _entry("b", 3, sets={"context.metadata.p": "B"})]
    assert replay_context(commit_ordered(entries, _completed("b", "a")), [])["metadata"]["p"] == "A"


def test_move_to_end_applies_only_to_the_steps_named_rerecorded():
    entries = [_entry(_SEED_STEP_ID, 1, adds={"context.metadata.l": ["a", "x"]}),
               _entry("s0", 2, adds={"context.metadata.l": ["a"]})]
    ordered = commit_ordered(entries, _completed("s0"))
    assert replay_context(ordered, [])["metadata"]["l"] == ["a", "x"]
    assert replay_context(ordered, ["s0"])["metadata"]["l"] == ["x", "a"]


# ── _union ───────────────────────────────────────────────────────────────────

def test_an_item_already_present_keeps_its_place():
    assert _union(["a", "b"], ["a"]) == ["a", "b"]


def test_new_items_append_in_order():
    assert _union(["a"], ["b", "c"]) == ["a", "b", "c"]


def test_a_re_record_moves_its_items_to_the_end():
    assert _union(["a", "b"], ["a"], move_to_end=True) == ["b", "a"]


def test_duplicates_within_one_write_collapse():
    assert _union([], ["a", "a", "b"]) == ["a", "b"]


# ── dotted paths ─────────────────────────────────────────────────────────────

def test_set_path_creates_the_parents_it_needs():
    doc: dict = {}
    _set_path(doc, "a.b.c", 1)
    assert doc == {"a": {"b": {"c": 1}}}


def test_set_path_replaces_a_non_dict_standing_where_a_parent_belongs():
    doc: dict = {"a": 5}
    _set_path(doc, "a.b", 1)
    assert doc == {"a": {"b": 1}}


def test_unset_path_is_a_no_op_when_the_path_is_absent():
    doc: dict = {"a": {"b": 1}}
    _unset_path(doc, "a.missing")
    _unset_path(doc, "nowhere.at.all")
    assert doc == {"a": {"b": 1}}
