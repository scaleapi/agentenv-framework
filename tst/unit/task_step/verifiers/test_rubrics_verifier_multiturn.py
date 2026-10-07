"""Unit tests for multi-turn trajectory handling in rubrics_verifier.

The verifier judges every turn of a multi-turn prompt_agent run, fed from the
per-turn URIs already on the PromptResponse (target_agent_per_turn_trajectory_object_urls):

- direct LLM judge  -> turns concatenated inline (``_merge_per_turn_text``)
- agent judge       -> turns loaded into a container dir (``_load_per_turn_trajectories``)

DEFAULT compaction runs per turn (externalized tool-result files namespaced ``turn_NN_``
so they don't collide); SCREENSHOT + multi-turn fails fast. These tests follow the
existing convention (test_rubrics_verifier_screenshots.py): patch the ``_read_trajectory_text``
S3 seam, stub ``_run_judge_with_output_retries`` to capture the judge input, and use an
AsyncMock sandbox for the loader — so nothing touches S3 or a real sandbox.
"""
from __future__ import annotations

import gc
import json
import weakref
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers import rubrics_verifier
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.config import get_config, set_object_store
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import CompactionType, TrajectoryFilter
from tst.unit.event_loop_probe import on_event_loop
from tst.unit.store.fakes import ConfiguredObjectStore


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
        target_agent_per_turn_trajectory_object_urls=["s3://t/1.json", "s3://t/2.json"],
        agent_trajectory_object_url="s3://t/2.json",
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
    pr = _pr(agent_trajectory_object_url="s3://a/last.json",
             target_agent_per_turn_trajectory_object_urls=["s3://a/1.json", "s3://a/last.json"])
    assert v._compact_trajectory(pr, None).container_object_url == "s3://a/last.json"


def test_compact_trajectory_falls_back_to_last_per_turn_uri():
    v = _verifier()
    pr = _pr(agent_trajectory_object_url=None,
             target_agent_per_turn_trajectory_object_urls=["s3://a/1.json", "s3://a/2.json"])
    assert v._compact_trajectory(pr, None).container_object_url == "s3://a/2.json"


def test_compact_trajectory_fallback_skips_none_entries():
    v = _verifier()
    pr = _pr(agent_trajectory_object_url=None,
             target_agent_per_turn_trajectory_object_urls=["s3://a/1.json", None])
    assert v._compact_trajectory(pr, None).container_object_url == "s3://a/1.json"


# --- gate / guard fail-fast (via execute; raises before judge/sandbox) --------

@pytest.mark.asyncio
async def test_screenshot_plus_multiturn_raises():
    v = _verifier(trajectory_filter=TrajectoryFilter(
        compaction_type=CompactionType.SCREENSHOT, screenshot_last_n=3))
    ctx = TaskStepContext(prompt_responses=[_pr(
        target_agent_per_turn_trajectory_object_urls=["s3://a/1.json", "s3://a/2.json"],
        agent_trajectory_object_url="s3://a/2.json",
    )])
    with pytest.raises(RuntimeError, match="SCREENSHOT"):
        await v.execute(ctx)


@pytest.mark.asyncio
async def test_guard_raises_when_all_trajectory_sources_empty():
    v = _verifier()
    ctx = TaskStepContext(prompt_responses=[_pr(
        target_agent_per_turn_trajectory_object_urls=[None], agent_trajectory_object_url=None,
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
        target_agent_per_turn_trajectory_object_urls=["s3://t/only.json"],
        agent_trajectory_object_url="s3://t/only.json",
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
    sandbox.write_file_from_object = AsyncMock()
    sandbox.write_file_from_text = AsyncMock()

    await v._load_per_turn_trajectories(
        sandbox, ["s3://a/1.json", "s3://a/2.json"], "/tmp/d", None)

    dests = [call.args[1] for call in sandbox.write_file_from_object.call_args_list]
    assert dests == ["/tmp/d/turn_01.json", "/tmp/d/turn_02.json"]
    sandbox.write_file_from_text.assert_not_called()


@pytest.mark.asyncio
async def test_load_per_turn_trajectories_default_filter_compacts_and_namespaces(monkeypatch):
    spans = json.dumps([_tool_span("t", {"result": "x" * 2000})])
    v = _verifier()
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: spans)
    sandbox = MagicMock()
    sandbox.write_file_from_object = AsyncMock()
    sandbox.write_file_from_text = AsyncMock()

    await v._load_per_turn_trajectories(sandbox, ["u1", "u2"], "/tmp/d", TrajectoryFilter())

    sandbox.write_file_from_object.assert_not_called()          # compacted -> written as text
    dests = [call.args[1] for call in sandbox.write_file_from_text.call_args_list]
    assert "/tmp/d/turn_01.json" in dests and "/tmp/d/turn_02.json" in dests
    tool_dests = [d for d in dests if "tool_call_result" in d]
    assert any("turn_01_tool_call_result" in d for d in tool_dests)
    assert any("turn_02_tool_call_result" in d for d in tool_dests)
    assert len(set(tool_dests)) == len(tool_dests)          # no cross-turn collisions


# --- trajectory reads and writes run off the event loop -----------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [1, 2], ids=["single-turn", "multi-turn"])
async def test_the_direct_judge_reads_trajectories_off_the_event_loop(monkeypatch, turns):
    reads: list[bool] = []
    v = _verifier()

    def read(uri):
        reads.append(on_event_loop())
        return "EVENTS"

    monkeypatch.setattr(v, "_read_trajectory_text", read)
    _capture_judge(monkeypatch, v)
    uris = [f"s3://t/{i}.json" for i in range(1, turns + 1)]
    ctx = TaskStepContext(prompt_responses=[_pr(target_agent_per_turn_trajectory_object_urls=uris, agent_trajectory_object_url=uris[-1])])

    await v.execute(ctx)

    assert len(reads) == turns and not any(reads)


class _LoopCheckingStore:
    """Serves one trajectory and records, per call, whether it ran on the event loop's thread."""

    def __init__(self, body: bytes) -> None:
        self.body, self.on_loop = body, []

    def get(self, object_url):
        self.on_loop.append(on_event_loop())
        return self.body

    def put(self, key, data, content_type=None, allow_overwrite=False):
        self.on_loop.append(on_event_loop())
        return f"s3://t/{key}"

    def object_url(self, key):
        return f"s3://t/{key}"


def _recording(monkeypatch, name, calls):
    """Patch ``rubrics_verifier.<name>`` to record whether each call ran on the loop's thread."""
    original = getattr(rubrics_verifier, name)

    def record(*args, **kwargs):
        calls.append(on_event_loop())
        return original(*args, **kwargs)

    monkeypatch.setattr(rubrics_verifier, name, record)


@pytest.mark.asyncio
async def test_default_compaction_reads_compacts_and_rewrites_off_the_event_loop(monkeypatch):
    store = _LoopCheckingStore(json.dumps([_tool_span("t", {"result": "x" * 2000})]).encode())
    set_object_store(store)
    compactions: list[bool] = []
    _recording(monkeypatch, "compact_otel_trajectory", compactions)
    v = _verifier(trajectory_filter=TrajectoryFilter())
    _capture_judge(monkeypatch, v)

    await v.execute(TaskStepContext(prompt_responses=[_pr(agent_trajectory_object_url="s3://t/raw.json")]))

    assert len(store.on_loop) >= 2 and not any(store.on_loop)
    assert compactions == [False]


@pytest.mark.asyncio
async def test_screenshot_compaction_reads_and_parses_off_the_event_loop(monkeypatch):
    shot = {"result": "ok", "screenshot": "/9j/" + "A" * 200}
    span = {**_tool_span("gui_screenshot", None), "attributes": {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.prompt": json.dumps({"tool": "gui_screenshot", "input": {}}),
        "gen_ai.completion": json.dumps(shot),
    }}
    store = _LoopCheckingStore(json.dumps([span]).encode())
    set_object_store(store)
    parses: list[bool] = []
    _recording(monkeypatch, "final_image_frames", parses)
    v = _verifier(trajectory_filter=TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT, screenshot_last_n=1))
    _capture_judge(monkeypatch, v)

    await v.execute(TaskStepContext(prompt_responses=[_pr(agent_trajectory_object_url="s3://t/raw.json")]))

    assert store.on_loop == [False] and parses == [False]


@pytest.mark.asyncio
async def test_per_turn_compaction_for_the_agent_judge_reads_and_compacts_off_the_event_loop(monkeypatch):
    reads: list[bool] = []
    compactions: list[bool] = []
    spans = json.dumps([_tool_span("t", {"result": "x" * 2000})])
    v = _verifier()

    def read(uri):
        reads.append(on_event_loop())
        return spans

    monkeypatch.setattr(v, "_read_trajectory_text", read)
    _recording(monkeypatch, "compact_otel_trajectory", compactions)
    sandbox = MagicMock()
    sandbox.write_file_from_text = AsyncMock()

    await v._load_per_turn_trajectories(sandbox, ["u1", "u2"], "/tmp/d", TrajectoryFilter())

    assert reads == [False, False] and compactions == [False, False]


class _Files(list):
    """A list a test can hold a weak reference to."""


@pytest.mark.asyncio
async def test_the_direct_judge_does_not_hold_the_externalized_tool_results_through_its_call(monkeypatch):
    set_object_store(_LoopCheckingStore(json.dumps([_tool_span("t", {"result": "x" * 2000})]).encode()))
    held: list[weakref.ref] = []
    compact = rubrics_verifier.compact_otel_trajectory

    def compact_recording(*args, **kwargs):
        filtered, files = compact(*args, **kwargs)
        files = _Files(files)
        held.append(weakref.ref(files))
        return filtered, files

    monkeypatch.setattr(rubrics_verifier, "compact_otel_trajectory", compact_recording)
    v = _verifier(trajectory_filter=TrajectoryFilter())
    alive: list[bool] = []

    async def judge(*, eval_prompt, **kwargs):
        gc.collect()
        alive.append(held[0]() is not None)
        return ([{**v.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(v, "_run_judge_with_output_retries", judge)

    await v.execute(TaskStepContext(prompt_responses=[_pr(agent_trajectory_object_url="s3://t/raw.json")]))

    assert held and alive == [False]


def test_a_trajectory_the_local_store_holds_is_read_from_it(cli_routing):
    """Read from the store holding the handed-in url; the compacted copy is a new object, minted in the
    configured store."""
    configured = ConfiguredObjectStore()
    set_object_store(configured)
    spans = [_tool_span("t", {"result": "x" * 2000})]
    url = get_config().get_object_store_for("@local/~/t").put("trajectories/raw.json", json.dumps(spans).encode())
    v = _verifier(trajectory_filter=TrajectoryFilter())

    assert json.loads(v._read_trajectory_text(url)) == spans
    compacted, _ = v._filter_trajectory(url, TrajectoryFilter())
    assert configured.owns(compacted)
