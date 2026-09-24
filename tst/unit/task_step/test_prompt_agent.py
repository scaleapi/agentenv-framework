"""prompt_agent: generic capture of an agent's StructuredOutput from its terminal response.

The agent harness flattens structured_output to a JSON string in the response text, so the
capture is a tolerant parse. It must be a no-op (None) for free-text replies, so the many
other prompt_agent users are unaffected."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import (
    PromptAgentTaskStep,
    _duplicates_prompt_text,
    _parse_structured_output,
)


def test_parses_pure_json_object():
    out = _parse_structured_output(
        '{"status":"success","artifacts":{"gold_patch":"/app/patches/gold_patch.diff"}}'
    )
    assert out["status"] == "success"
    assert out["artifacts"]["gold_patch"].endswith("gold_patch.diff")


def test_strips_json_code_fence():
    assert _parse_structured_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_structured_output('```json{"a": 1}```') == {"a": 1}


def test_none_for_non_object_responses():
    # free text, a JSON array (not an object), and empty/whitespace all yield None -> no capture
    assert _parse_structured_output("I refactored the repo and zipped the harbor.") is None
    assert _parse_structured_output("[1, 2, 3]") is None
    assert _parse_structured_output("") is None
    assert _parse_structured_output("   \n  ") is None


def test_tolerates_surrounding_whitespace():
    assert _parse_structured_output('  \n {"x": true}\n ') == {"x": True}


def test_extracts_object_embedded_in_prose():
    # the harness may emit the structured_output alongside other text blocks
    assert _parse_structured_output('Done. Here is the result:\n{"status": "success"}') == {"status": "success"}
    assert _parse_structured_output('{"status": "success"}\n\nLet me know if you need anything else.') == {"status": "success"}
    # nested objects survive (raw_decode balances braces)
    assert _parse_structured_output('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}
    # an earlier JSON-looking snippet in prose must NOT win over the real (last) output
    assert _parse_structured_output(
        'considered {"foo": 1} but produced:\n{"status": "ok"}'
    ) == {"status": "ok"}


# _duplicates_prompt_text: turn-0 dedup of source_agent_per_turn_prompt_parts vs
# prompt_text. Only the exact prompt-mode shape — a single text part whose
# text equals prompt_text — is a duplicate; anything else must be stored verbatim.


def test_duplicate_when_single_text_part_equals_prompt_text():
    prompt = "Fix the bug in /inputs/main.py\n\ndata:image/png;base64,iVBORw0KGgo..."
    assert _duplicates_prompt_text([{"kind": "text", "text": prompt}], prompt) is True


def test_not_duplicate_without_prompt_text():
    # parts-mode steps persist prompt_text=None — nothing to dedup against
    assert _duplicates_prompt_text([{"kind": "text", "text": "hello"}], None) is False


def test_not_duplicate_for_different_text():
    assert _duplicates_prompt_text([{"kind": "text", "text": "hello"}], "other") is False


def test_not_duplicate_for_multi_part_or_non_text_parts():
    prompt = "hello"
    multi = [
        {"kind": "text", "text": prompt},
        {"kind": "file", "file": {"uri": "s3://bucket/a.png"}},
    ]
    assert _duplicates_prompt_text(multi, prompt) is False
    assert _duplicates_prompt_text([{"kind": "file", "file": {"uri": "s3://b/a.png"}}], prompt) is False
    assert _duplicates_prompt_text([], prompt) is False


def test_not_duplicate_when_part_carries_extra_keys():
    # extra metadata on the part means it is NOT the materialized prompt shape
    assert (
        _duplicates_prompt_text([{"kind": "text", "text": "hello", "metadata": {"a": 1}}], "hello")
        is False
    )


def test_fetch_trajectory_object_named_by_target_a2a_task_id(monkeypatch):
    """Inline trajectory is named by `target_a2a_task_id` (the client message id, recorded on the
    conversation as `a2a_task_id`), not `a2a_server_task_id` — so it's resolvable from a conversation."""
    client = MagicMock()
    client.__aenter__.return_value.post = AsyncMock(
        return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"trajectory": {}})
    )
    monkeypatch.setattr(pa.httpx, "AsyncClient", lambda *a, **k: client)
    store = SimpleNamespace(get_object_key=lambda p: "traj/", put=lambda key, *a, **k: f"s3://b/{key}")
    # Patched on `agent_env.config`, not on `pa`: the upload now runs in
    # `agent_state_capture.upload_trajectory` (shared with the periodic capture),
    # which resolves `get_config` from the config module at call time.
    import agent_env.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_config", lambda: SimpleNamespace(get_object_store=lambda: store)
    )

    # unbound call: the inline branch never touches `self`, so we skip AWS-touching construction.
    uri = asyncio.run(PromptAgentTaskStep._fetch_trajectory(
        object(), "http://a", {"params": {"endpoint": "/t"}}, "SERVER-ID", "s3://b/traj/", "CLIENT-ID"))
    assert uri == "s3://b/traj/trajectory-CLIENT-ID.json"
