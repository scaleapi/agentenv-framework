"""ContextUpdateOps as a journal record: Mongo-safe encoding round-trips, a
duplicate list append is a genuine no-op, nested list items are redaction-stripped, and a
context rebuilt from the store gets its secrets regrafted."""

from __future__ import annotations

from agent_env.task_step.context import TaskStepContext, regraft_redacted_keys
from agent_env.task_step.context_ops import ContextUpdateOps, build_context_update_ops


def test_journal_dict_round_trips_and_has_no_dotted_keys():
    pre = TaskStepContext(metadata={"a": 1, "l": [1], "gone": 1})
    post = TaskStepContext(metadata={"a": 2, "l": [1, 2], "n": {"x": 1}}, agent_model="m")
    ops = build_context_update_ops(pre, post)
    d = ops.to_journal_dict()

    def _no_dotted_keys(v):
        if isinstance(v, dict):
            assert all("." not in k for k in v), v
            for x in v.values():
                _no_dotted_keys(x)
        elif isinstance(v, list):
            for x in v:
                _no_dotted_keys(x)
    _no_dotted_keys(d)
    assert ContextUpdateOps.from_journal_dict(d) == ops
    assert "context.metadata.gone" in ops.unsets and ops.sets["context.metadata.a"] == 2


def test_appending_an_item_the_list_already_holds_emits_nothing():
    assert build_context_update_ops(TaskStepContext(metadata={"k": [1]}),
                                    TaskStepContext(metadata={"k": [1, 1]})).is_empty()
    ops = build_context_update_ops(TaskStepContext(metadata={"k": [1, 2]}),
                                   TaskStepContext(metadata={"k": [1, 2, 2, 3]}))
    assert ops.add_to_sets == {"context.metadata.k": [3]}


def test_nested_list_items_are_redaction_stripped():
    pre = TaskStepContext(metadata={"rows": [{"id": 1, "remote_tokens": "t0"}]})
    post = TaskStepContext(metadata={"rows": [{"id": 1, "remote_tokens": "t0"}, {"id": 2, "remote_tokens": "t1"}],
                                     "fresh": [{"id": 3, "litellm_api_key": "sk"}]})
    ops = build_context_update_ops(pre, post)
    blob = repr(ops.to_journal_dict())
    assert "t1" not in blob and "sk" not in blob
    assert ops.add_to_sets["context.metadata.rows"] == [{"id": 2}]


def test_regraft_redacted_keys_restores_secrets_into_a_stored_context():
    live = {"user_overrides": {"litellm_api_key": "sk", "priority": 1}, "remote_tokens": "r", "plain": 1}
    stored = {"user_overrides": {"priority": 1}, "plain": 1}
    regraft_redacted_keys(live, stored)
    assert stored == live


def test_regraft_rebuilds_a_parent_dict_the_rebuild_dropped_entirely():
    # A dict holding nothing but secrets leaves no trace in a stored doc or a journal diff,
    # so the rebuilt context has no `auth` key at all — the secret must still land.
    live = {"auth": {"remote_tokens": "tok"}, "plain": 1}
    stored = {"plain": 1}
    regraft_redacted_keys(live, stored)
    assert stored == {"auth": {"remote_tokens": "tok"}, "plain": 1}


def test_regraft_rebuilds_nested_parents_but_grafts_no_secretless_dict():
    live = {"a": {"b": {"cf_access_client_secret": "s"}}, "empty": {"x": 1}}
    stored = {}
    regraft_redacted_keys(live, stored)
    assert stored == {"a": {"b": {"cf_access_client_secret": "s"}}}   # `empty` carries no secret: not grafted


def test_regraft_never_overwrites_a_value_the_rebuild_did_write():
    live = {"auth": {"remote_tokens": "tok"}}
    stored = {"auth": "a-real-scalar"}
    regraft_redacted_keys(live, stored)
    assert stored == {"auth": "a-real-scalar"}
