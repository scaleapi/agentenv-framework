"""Unit tests for multi-turn trajectory handling in rubrics_verifier.

The verifier judges every turn of a multi-turn prompt_agent run, fed from the
per-turn URIs already on the PromptResponse (target_agent_per_turn_trajectory_s3_uris):

- direct LLM judge  -> turns concatenated inline (``_merge_per_turn_text``)
- agent judge       -> turns loaded into a container dir (``_load_per_turn_trajectories``)

DEFAULT compaction runs per turn (externalized tool-result files namespaced ``turn_NN_``
so they don't collide); SCREENSHOT + multi-turn fails fast. These tests follow the
existing convention (test_rubrics_verifier_screenshots.py): patch the ``_read_trajectory_text``
S3 seam, stub ``_run_judge_with_output_retries`` to capture the judge input, and use an
AsyncMock sandbox for the loader — so nothing touches S3 or a real sandbox.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import CompactionType, TrajectoryFilter


def _verifier(**kw) -> RubricsVerifierTaskStep:
    params = dict(
        id="verify", version=1,
        criteria=[{"id": "c1", "description": "outcome"}],
        prompt_id="p1", use_agent_judge=False, use_trajectory=True,
        verifier_id="vt",
    )
    params.update(kw)
    return RubricsVerifierTaskStep(**params)


def _tool_span(tool: str, output) -> dict:
    """A raw execute_tool OTel span; large ``output`` triggers externalization."""
    return {
        "name": tool,
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.prompt": json.dumps({"tool": tool, "input": {}}),
            "gen_ai.completion": json.dumps({"output": output}),
        },
        "start_time": "1",
        "end_time": "2",
    }


def _capture_judge(monkeypatch, verifier) -> dict:
    """Stub the judge call and capture the prompt it would receive."""
    captured: dict = {}

    async def fake_run(*, eval_prompt, **kwargs):
        captured["eval_prompt"] = eval_prompt
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)
    return captured


def _pr(**kw) -> PromptResponse:
    params = dict(prompt_id="p1", response="done", prompt_text="do it")
    params.update(kw)
    return PromptResponse(**params)


# --- direct-LLM judge: multi-turn inline delivery (via execute) ---------------

@pytest.mark.asyncio
async def test_direct_judge_multiturn_inlines_all_turns_marked(monkeypatch):
    fixtures = {"s3://t/1.json": "TURN_ONE_EVENTS", "s3://t/2.json": "TURN_TWO_EVENTS"}
    v = _verifier()
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: fixtures[uri])
    captured = _capture_judge(monkeypatch, v)
    ctx = TaskStepContext(prompt_responses=[_pr(
        target_agent_per_turn_trajectory_s3_uris=["s3://t/1.json", "s3://t/2.json"],
        agent_trajectory_s3_uri="s3://t/2.json",
    )])

    await v.execute(ctx)

    prompt = captured["eval_prompt"]
    assert "===== Turn 1 =====" in prompt and "===== Turn 2 =====" in prompt
    assert "TURN_ONE_EVENTS" in prompt and "TURN_TWO_EVENTS" in prompt
    assert prompt.index("TURN_ONE_EVENTS") < prompt.index("TURN_TWO_EVENTS")
    assert ctx.metadata["verifications"]["vt"]["score"] == 1.0


# --- _merge_per_turn_text logic ----------------------------------------------

def test_merge_per_turn_text_default_filter_compacts_each_turn(monkeypatch):
    fixtures = {
        "u1": json.dumps([_tool_span("email_send", {"result": "x" * 2000})]),   # externalized
        "u2": json.dumps([_tool_span("reminder_move", {"ok": True})]),          # inlined
    }
    v = _verifier()
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: fixtures[uri])
    out = v._merge_per_turn_text(["u1", "u2"], TrajectoryFilter())
    assert "===== Turn 1 =====" in out and "===== Turn 2 =====" in out
    assert "email_send" in out and "reminder_move" in out
    assert "output_file" in out            # turn-1 large result externalized (reference kept)
    assert "x" * 2000 not in out           # ...raw blob not inlined


# --- _compact_trajectory raw_uri fallback (no-filter path does no S3 read) -----

def test_compact_trajectory_prefers_agent_trajectory_uri():
    v = _verifier()
    pr = _pr(agent_trajectory_s3_uri="s3://a/last.json",
             target_agent_per_turn_trajectory_s3_uris=["s3://a/1.json", "s3://a/last.json"])
    assert v._compact_trajectory(pr, None).container_s3_uri == "s3://a/last.json"


def test_compact_trajectory_falls_back_to_last_per_turn_uri():
    v = _verifier()
    pr = _pr(agent_trajectory_s3_uri=None,
             target_agent_per_turn_trajectory_s3_uris=["s3://a/1.json", "s3://a/2.json"])
    assert v._compact_trajectory(pr, None).container_s3_uri == "s3://a/2.json"


def test_compact_trajectory_fallback_skips_none_entries():
    v = _verifier()
    pr = _pr(agent_trajectory_s3_uri=None,
             target_agent_per_turn_trajectory_s3_uris=["s3://a/1.json", None])
    assert v._compact_trajectory(pr, None).container_s3_uri == "s3://a/1.json"


# --- gate / guard fail-fast (via execute; raises before judge/sandbox) --------

@pytest.mark.asyncio
async def test_screenshot_plus_multiturn_raises():
    v = _verifier(trajectory_filter=TrajectoryFilter(
        compaction_type=CompactionType.SCREENSHOT, screenshot_last_n=3))
    ctx = TaskStepContext(prompt_responses=[_pr(
        target_agent_per_turn_trajectory_s3_uris=["s3://a/1.json", "s3://a/2.json"],
        agent_trajectory_s3_uri="s3://a/2.json",
    )])
    with pytest.raises(RuntimeError, match="SCREENSHOT"):
        await v.execute(ctx)


@pytest.mark.asyncio
async def test_guard_raises_when_all_trajectory_sources_empty():
    v = _verifier()
    ctx = TaskStepContext(prompt_responses=[_pr(
        target_agent_per_turn_trajectory_s3_uris=[None], agent_trajectory_s3_uri=None,
    )])
    with pytest.raises(RuntimeError, match="No trajectory available"):
        await v.execute(ctx)


@pytest.mark.asyncio
async def test_single_turn_does_not_take_multiturn_path(monkeypatch):
    # Exactly one non-None per-turn URI must route to the single-file path, not the
    # multi-turn merge — guards the ``len(...) >= 2`` gate against an off-by-one to >= 1.
    v = _verifier()
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: "SINGLE_TURN_EVENTS")

    def _no_merge(*a, **k):
        raise AssertionError("multi-turn merge must not run for a single-turn response")

    monkeypatch.setattr(v, "_merge_per_turn_text", _no_merge)
    captured = _capture_judge(monkeypatch, v)
    ctx = TaskStepContext(prompt_responses=[_pr(
        target_agent_per_turn_trajectory_s3_uris=["s3://t/only.json"],
        agent_trajectory_s3_uri="s3://t/only.json",
    )])

    await v.execute(ctx)

    prompt = captured["eval_prompt"]
    assert "===== Turn" not in prompt        # single-file path (no per-turn markers)
    assert "SINGLE_TURN_EVENTS" in prompt     # the one trajectory was inlined


# --- _load_per_turn_trajectories: container delivery call-sequence -------------

@pytest.mark.asyncio
async def test_load_per_turn_trajectories_no_filter_writes_each_turn():
    v = _verifier()
    sandbox = MagicMock()
    sandbox.write_file_from_s3 = AsyncMock()
    sandbox.write_file_from_text = AsyncMock()

    await v._load_per_turn_trajectories(
        sandbox, ["s3://a/1.json", "s3://a/2.json"], "/tmp/d", None)

    dests = [call.args[1] for call in sandbox.write_file_from_s3.call_args_list]
    assert dests == ["/tmp/d/turn_01.json", "/tmp/d/turn_02.json"]
    sandbox.write_file_from_text.assert_not_called()


@pytest.mark.asyncio
async def test_load_per_turn_trajectories_default_filter_compacts_and_namespaces(monkeypatch):
    spans = json.dumps([_tool_span("t", {"result": "x" * 2000})])
    v = _verifier()
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: spans)
    sandbox = MagicMock()
    sandbox.write_file_from_s3 = AsyncMock()
    sandbox.write_file_from_text = AsyncMock()

    await v._load_per_turn_trajectories(sandbox, ["u1", "u2"], "/tmp/d", TrajectoryFilter())

    sandbox.write_file_from_s3.assert_not_called()          # compacted -> written as text
    dests = [call.args[1] for call in sandbox.write_file_from_text.call_args_list]
    assert "/tmp/d/turn_01.json" in dests and "/tmp/d/turn_02.json" in dests
    tool_dests = [d for d in dests if "tool_call_result" in d]
    assert any("turn_01_tool_call_result" in d for d in tool_dests)
    assert any("turn_02_tool_call_result" in d for d in tool_dests)
    assert len(set(tool_dests)) == len(tool_dests)          # no cross-turn collisions
