"""Unit tests for context_ops — path-level diff + UpdateSpec compilation."""

import copy
import dataclasses

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.context_ops import (
    ContextUpdateOps,
    build_context_update_ops,
)


def _env(instance_id: str = "i1") -> DeployedEnv:
    return DeployedGatewayEnv(
        env_id="e", env_version=1, gateway_url="g", mcp_url="m",
        db_web_url=None, sandbox_id="s", instance_id=instance_id,
    )


def _agent(sandbox_id: str = "s1") -> DeployedAgent:
    return DeployedAgent(agent_name="a", api_url="u", sandbox_id=sandbox_id)


def _response(prompt_id: str = "p1") -> PromptResponse:
    return PromptResponse(prompt_id=prompt_id, response="hi")


def test_prompt_response_error_code_round_trips() -> None:
    response = PromptResponse.from_dict({
        "prompt_id": "p1",
        "response": "retry later",
        "error_type": "agent_error",
        "error_code": "provider_rate_limited",
    })

    assert response.error_code == "provider_rate_limited"
    assert dataclasses.asdict(response)["error_code"] == "provider_rate_limited"


def test_ops_is_empty_when_no_fields_populated():
    assert ContextUpdateOps().is_empty()


def test_ops_is_not_empty_when_any_field_populated():
    assert not ContextUpdateOps(sets={"x": 1}).is_empty()
    assert not ContextUpdateOps(unsets={"x"}).is_empty()
    assert not ContextUpdateOps(add_to_sets={"x": [1]}).is_empty()


def test_identity_produces_empty_ops():
    ctx = TaskStepContext(metadata={"k": "v"})
    ops = build_context_update_ops(ctx, ctx)
    assert ops.is_empty()


def test_pre_none_and_empty_post_produces_empty_ops():
    assert build_context_update_ops(None, TaskStepContext()).is_empty()


def test_pre_none_emits_all_populated_fields():
    post = TaskStepContext(
        deployed_envs=[_env()],
        agent_model="gpt-4",
        metadata={"k": "v"},
    )
    ops = build_context_update_ops(None, post)
    assert ops.sets == {"context.agent_model": "gpt-4", "context.metadata.k": "v"}
    assert "context.deployed_envs" in ops.add_to_sets
    assert len(ops.add_to_sets["context.deployed_envs"]) == 1


def test_new_list_item_goes_to_add_to_set():
    pre = TaskStepContext()
    post = TaskStepContext(deployed_envs=[_env("i1")])
    ops = build_context_update_ops(pre, post)
    assert "context.deployed_envs" in ops.add_to_sets
    assert ops.add_to_sets["context.deployed_envs"][0]["instance_id"] == "i1"


def test_duplicate_list_item_produces_no_op():
    env = _env("i1")
    pre = TaskStepContext(deployed_envs=[env])
    post = TaskStepContext(deployed_envs=[env])
    ops = build_context_update_ops(pre, post)
    assert "context.deployed_envs" not in ops.add_to_sets


def test_only_new_items_flow_to_add_to_set():
    pre = TaskStepContext(deployed_envs=[_env("i1")])
    post = TaskStepContext(deployed_envs=[_env("i1"), _env("i2")])
    ops = build_context_update_ops(pre, post)
    emitted = ops.add_to_sets["context.deployed_envs"]
    assert len(emitted) == 1
    assert emitted[0]["instance_id"] == "i2"


def test_agents_and_responses_lists_use_add_to_set():
    pre = TaskStepContext()
    post = TaskStepContext(
        deployed_agents=[_agent("s1")],
        prompt_responses=[_response("p1")],
    )
    ops = build_context_update_ops(pre, post)
    assert "context.deployed_agents" in ops.add_to_sets
    assert "context.prompt_responses" in ops.add_to_sets


@pytest.mark.parametrize("append_new", [False, True])
def test_response_history_diff_uses_linear_comparisons(monkeypatch, append_new):
    pre = TaskStepContext(prompt_responses=[_response(f"p{i}") for i in range(100)])
    post = copy.deepcopy(pre)
    if append_new:
        post.prompt_responses.append(_response("new"))

    comparisons = 0
    original_eq = PromptResponse.__eq__

    def count_equal(self, other):
        nonlocal comparisons
        comparisons += 1
        return original_eq(self, other)

    monkeypatch.setattr(PromptResponse, "__eq__", count_equal)
    ops = build_context_update_ops(pre, post)

    if append_new:
        assert ops.add_to_sets["context.prompt_responses"] == [dataclasses.asdict(_response("new"))]
    else:
        assert ops.is_empty()
    assert comparisons <= 2 * len(pre.prompt_responses)


def test_scalar_nil_to_value_emits_set():
    pre = TaskStepContext()
    post = TaskStepContext(agent_model="gpt-4")
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.agent_model": "gpt-4"}


def test_scalar_value_to_nil_emits_unset():
    pre = TaskStepContext(agent_model="gpt-4")
    post = TaskStepContext()
    ops = build_context_update_ops(pre, post)
    assert ops.unsets == {"context.agent_model"}


def test_scalar_unchanged_emits_nothing():
    pre = TaskStepContext(agent_model="gpt-4")
    post = TaskStepContext(agent_model="gpt-4")
    ops = build_context_update_ops(pre, post)
    assert ops.is_empty()


def test_scalar_changed_emits_set():
    pre = TaskStepContext(agent_model="gpt-4")
    post = TaskStepContext(agent_model="claude-4")
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.agent_model": "claude-4"}


def test_nested_sibling_leaf_addition_does_not_rewrite_parent():
    """Adding a nested key must emit only the new leaf, never the whole parent subtree."""
    pre = TaskStepContext(metadata={"verifications": {"A": {"score": 1.0}}})
    post = TaskStepContext(metadata={
        "verifications": {"A": {"score": 1.0}, "B": {"score": 0.5}},
    })
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.verifications.B.score": 0.5}
    assert "context.metadata.verifications" not in ops.sets


def test_new_nested_dict_recurses_to_leaves():
    """Recurse to distinct leaf paths so concurrent writers under a new parent don't clobber each other."""
    pre = TaskStepContext()
    post = TaskStepContext(metadata={"verifications": {"A": {"score": 1.0}}})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.verifications.A.score": 1.0}


def test_nested_deep_leaf_change_emits_full_leaf_path():
    pre = TaskStepContext(metadata={"a": {"b": {"c": 1}}})
    post = TaskStepContext(metadata={"a": {"b": {"c": 2}}})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.a.b.c": 2}


def test_top_level_metadata_key_deletion_emits_unset():
    pre = TaskStepContext(metadata={"foo": 1, "bar": 2})
    post = TaskStepContext(metadata={"bar": 2})
    ops = build_context_update_ops(pre, post)
    assert ops.unsets == {"context.metadata.foo"}


def test_nested_metadata_key_deletion_emits_unset_at_leaf_path():
    pre = TaskStepContext(metadata={"a": {"b": 1, "c": 2}})
    post = TaskStepContext(metadata={"a": {"b": 1}})
    ops = build_context_update_ops(pre, post)
    assert ops.unsets == {"context.metadata.a.c"}


def test_subtree_replaced_with_scalar_emits_set_at_path():
    pre = TaskStepContext(metadata={"k": {"nested": 1}})
    post = TaskStepContext(metadata={"k": "scalar"})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.k": "scalar"}


def test_scalar_replaced_with_subtree_emits_set_at_path():
    pre = TaskStepContext(metadata={"k": "scalar"})
    post = TaskStepContext(metadata={"k": {"nested": 1}})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.k": {"nested": 1}}


def test_metadata_list_append_emits_add_to_set_with_tail():
    pre = TaskStepContext(metadata={"runs": [{"id": "a"}]})
    post = TaskStepContext(metadata={"runs": [{"id": "a"}, {"id": "b"}]})
    ops = build_context_update_ops(pre, post)
    assert ops.add_to_sets == {"context.metadata.runs": [{"id": "b"}]}
    assert not ops.sets


def test_metadata_new_list_key_emits_add_to_set():
    """First creator of a list also uses $addToSet, so concurrent first-creators get deduped."""
    pre = TaskStepContext()
    post = TaskStepContext(metadata={"runs": [{"id": "a"}]})
    ops = build_context_update_ops(pre, post)
    assert ops.add_to_sets == {"context.metadata.runs": [{"id": "a"}]}
    assert not ops.sets


def test_metadata_list_reordered_falls_back_to_set():
    pre = TaskStepContext(metadata={"runs": [1, 2]})
    post = TaskStepContext(metadata={"runs": [2, 1]})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.runs": [2, 1]}
    assert not ops.add_to_sets


def test_metadata_list_shortened_falls_back_to_set():
    pre = TaskStepContext(metadata={"runs": [1, 2]})
    post = TaskStepContext(metadata={"runs": [1]})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.runs": [1]}


def test_metadata_list_unchanged_emits_nothing():
    pre = TaskStepContext(metadata={"runs": [1, 2]})
    post = TaskStepContext(metadata={"runs": [1, 2]})
    ops = build_context_update_ops(pre, post)
    assert ops.is_empty()


def test_redacted_top_level_metadata_key_not_emitted():
    pre = TaskStepContext()
    post = TaskStepContext(metadata={"usersim_api_key": "secret", "ok": "yes"})
    ops = build_context_update_ops(pre, post)
    assert "context.metadata.usersim_api_key" not in ops.sets
    assert ops.sets == {"context.metadata.ok": "yes"}


def test_redacted_key_inside_user_overrides_filtered_from_set():
    pre = TaskStepContext()
    post = TaskStepContext(metadata={
        "user_overrides": {"litellm_api_key": "x", "model": "claude"},
    })
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.user_overrides.model": "claude"}


def test_cf_access_secret_redacted_but_url_kept():
    pre = TaskStepContext()
    post = TaskStepContext(metadata={
        "user_overrides": {
            "env_url": "https://env-3.example.com",
            "cf_access_client_id": "client-id",
            "cf_access_client_secret": "shhh",
        },
    })
    ops = build_context_update_ops(pre, post)
    assert "context.metadata.user_overrides.cf_access_client_secret" not in ops.sets
    assert ops.sets == {
        "context.metadata.user_overrides.env_url": "https://env-3.example.com",
        "context.metadata.user_overrides.cf_access_client_id": "client-id",
    }


def test_redacted_key_at_deep_path_filtered():
    pre = TaskStepContext()
    post = TaskStepContext(metadata={
        "deep": {"nested": {"litellm_api_key": "x", "keep": 1}},
    })
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata.deep.nested.keep": 1}


def test_redacted_key_changed_still_emits_nothing():
    pre = TaskStepContext(metadata={"usersim_api_key": "old"})
    post = TaskStepContext(metadata={"usersim_api_key": "new"})
    ops = build_context_update_ops(pre, post)
    assert ops.is_empty()


def test_dotted_metadata_key_falls_back_to_wholesale_set():
    """A dotted root key forces a wholesale $set of context.metadata (dots aren't valid path segments)."""
    pre = TaskStepContext()
    post = TaskStepContext(metadata={"a.b": 1, "ok": 2})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata": {"a.b": 1, "ok": 2}}
    assert ops.unsets == set()
    assert ops.add_to_sets == {}


def test_dollar_prefixed_metadata_key_falls_back_to_wholesale_set():
    pre = TaskStepContext()
    post = TaskStepContext(metadata={"$where": 1})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {"context.metadata": {"$where": 1}}


def test_dotted_key_deep_in_metadata_falls_back_at_nearest_path():
    """The fallback is scoped to the dirty subtree; clean siblings still get path-level ops."""
    pre = TaskStepContext(metadata={"outer": {}, "sibling": "old"})
    post = TaskStepContext(metadata={"outer": {"a.b": 1}, "sibling": "new"})
    ops = build_context_update_ops(pre, post)
    assert ops.sets["context.metadata.outer"] == {"a.b": 1}
    assert ops.sets["context.metadata.sibling"] == "new"


def test_filename_keyed_output_urls_falls_back_to_wholesale_set():
    """Filename keys (with dots) once raised; now they $set the whole dict (output_urls repro)."""
    pre = TaskStepContext(metadata={"output_urls": {}})
    post = TaskStepContext(metadata={"output_urls": {
        "trajectory.json": "s3://bucket/run/trajectory.json",
        "conversation_log.jsonl": "s3://bucket/run/conversation_log.jsonl",
        "done.flag": "s3://bucket/run/done.flag",
    }})
    ops = build_context_update_ops(pre, post)
    assert ops.sets == {
        "context.metadata.output_urls": {
            "trajectory.json": "s3://bucket/run/trajectory.json",
            "conversation_log.jsonl": "s3://bucket/run/conversation_log.jsonl",
            "done.flag": "s3://bucket/run/done.flag",
        }
    }
    assert ops.unsets == set()


def _apply_ops_to_mirror(doc: dict, ops: ContextUpdateOps) -> dict:
    """Apply ops to an in-memory dict, modeling Mongo's $set / $unset / $addToSet."""
    out = copy.deepcopy(doc)

    def descend(d: dict, path: list[str]) -> dict:
        for seg in path[:-1]:
            d = d.setdefault(seg, {})
        return d

    for p, v in ops.sets.items():
        parts = p.split(".")
        descend(out, parts)[parts[-1]] = v
    for p in ops.unsets:
        parts = p.split(".")
        parent = out
        for seg in parts[:-1]:
            parent = parent.get(seg, {})
        parent.pop(parts[-1], None)
    for p, items in ops.add_to_sets.items():
        parts = p.split(".")
        parent = descend(out, parts)
        arr = parent.setdefault(parts[-1], [])
        for item in items:
            if item not in arr:
                arr.append(item)
    return out


def test_in_place_mutation_of_appended_list_item_duplicates_in_doc():
    """Regression: mutating an item already present in a top-level $addToSet list
    re-adds it as a NEW element instead of updating in place.

    Reproduces the production bug where ``rubrics_verifier`` set the compacted
    trajectory's URL on a ``prompt_response`` that ``prompt_agent``
    had already appended to ``context.prompt_responses``. The mutated item is
    BSON-unequal to the stored one, so the diff emits it via ``$addToSet`` and
    the instance ends up with two near-identical prompt_responses — the hub then
    renders the same trajectory twice. Items in append-only context lists MUST be
    immutable once appended; record post-append data elsewhere (e.g. the verifier
    result). See rubrics_verifier.py.
    """
    pr = _response("p1")
    pre = TaskStepContext(prompt_responses=[pr])

    # Model the (now-fixed) in-place mutation: same logical item, one field changed.
    mutated = _response("p1")
    mutated.tool_call_count = (mutated.tool_call_count or 0) + 1
    post = TaskStepContext(prompt_responses=[mutated])

    ops = build_context_update_ops(pre, post)
    assert "context.prompt_responses" in ops.add_to_sets

    # Applied on top of a doc that already holds the pre item, this yields TWO.
    doc = {"context": {"prompt_responses": [dataclasses.asdict(pr)]}}
    out = _apply_ops_to_mirror(doc, ops)
    assert len(out["context"]["prompt_responses"]) == 2  # the duplication

    # Guard the fix: leaving the appended item untouched emits no new element.
    unchanged = TaskStepContext(prompt_responses=[_response("p1")])
    clean_ops = build_context_update_ops(pre, unchanged)
    assert "context.prompt_responses" not in clean_ops.add_to_sets


def test_ops_idempotent_under_double_apply():
    pre = TaskStepContext(metadata={"a": 1})
    post = TaskStepContext(
        deployed_envs=[_env("i1")],
        metadata={"a": 1, "verifications": {"A": {"score": 1.0}}, "runs": [{"id": "r1"}]},
    )
    ops = build_context_update_ops(pre, post)
    initial: dict = {"context": {"metadata": {"a": 1}}}
    once = _apply_ops_to_mirror(initial, ops)
    twice = _apply_ops_to_mirror(once, ops)
    assert once == twice
