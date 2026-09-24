"""Tests for opt-in screenshot grading in rubrics_verifier.

Covers the frame extractor, the original bug (extracting from the COMPACTED
trajectory finds zero frames because compaction externalizes screenshots), and
the execute() wiring (frames come from the RAW trajectory while the judge's text
stays the compacted trajectory).
"""
from __future__ import annotations

import json

import pytest

from agent_env.config.model import ModelConfig
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import SCREENSHOT_GROUNDING
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    RubricsVerifierTaskStep,
    _image_url_block,
    _looks_like_response_format_error,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    CompactionType,
    ImageFrame,
    TrajectoryFilter,
    media_type_for_b64,
    compact_otel_trajectory,
    compact_screenshot_trajectory,
    final_frames_from_raw,
    strip_base64_images,
)


def _expected_block(b64: str) -> dict:
    """The litellm image block the verifier should produce for a raw base64 frame."""
    return _image_url_block(ImageFrame(media_type_for_b64(b64), b64))

# A real 1x1 PNG (magic bytes -> base64 starts "iVBORw0KGgo"). claude_cua encodes
# screenshots as PNG, so the media type must be inferred, not hardcoded to jpeg.
_PNG_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)


def _screenshot_filter(last_n: int = 3) -> TrajectoryFilter:
    return TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT, screenshot_last_n=last_n)

_frames = final_frames_from_raw


def _shot_span(screenshot: str, *, completion_as_dict: bool = False) -> dict:
    """A raw execute_tool span shaped like the iOS CUA bridge emits."""
    comp = {"result": "ok", "screenshot": screenshot}
    return {
        "name": "ios_screenshot",
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.prompt": json.dumps({"tool": "ios_screenshot", "input": {}}),
            "gen_ai.completion": comp if completion_as_dict else json.dumps(comp),
        },
        "start_time": "1",
        "end_time": "2",
    }


def test_screenshot_grounding_accepts_intermediate_backend_and_process_alias():
    grounding = SCREENSHOT_GROUNDING.format(n=1)

    assert '"backend": "intermediate"' in grounding
    assert '"backend": "process"' in grounding
    assert 'deprecated alias' in grounding
    assert 'trajectory text / action log' in grounding


# --------------------------------------------------------------------------- #
# _final_frames_from_raw — pure helper

def test_returns_last_n_base64_frames():
    raw = json.dumps([_shot_span("AAAA"), _shot_span("BBBB"), _shot_span("CCCC")])
    assert _frames(raw, last_n=2) == ["BBBB", "CCCC"]  # raw base64, no data: prefix


def test_handles_completion_stored_as_dict():
    raw = json.dumps([_shot_span("ZZZZ", completion_as_dict=True)])
    assert _frames(raw, last_n=3) == ["ZZZZ"]


def test_last_n_non_positive_returns_nothing():
    # Non-positive must NOT mean "attach all" — that's a footgun for long trajectories.
    raw = json.dumps([_shot_span("AAAA"), _shot_span("BBBB")])
    assert _frames(raw, last_n=0) == []
    assert _frames(raw, last_n=-1) == []


def test_no_screenshots_returns_empty():
    raw = json.dumps([{"attributes": {"gen_ai.completion": json.dumps({"result": "ok"})}}])
    assert _frames(raw, last_n=3) == []


def test_non_list_payload_returns_empty():
    assert _frames(json.dumps({"not": "a list"}), last_n=3) == []


def test_malformed_trajectory_is_a_safe_noop():
    assert _frames("not json at all", last_n=3) == []


def test_media_type_inferred_from_magic_bytes():
    # JPEG magic → image/jpeg; PNG magic (real fixture) → image/png; unknown → png default.
    assert media_type_for_b64("/9j/AAAA") == "image/jpeg"
    assert media_type_for_b64(_PNG_1x1) == "image/png"
    assert media_type_for_b64("QUJD") == "image/png"


def test_image_url_block_adapts_image_frame():
    block = _image_url_block(ImageFrame("image/png", "QUJD"))
    assert block == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def test_compact_screenshot_infers_png_media_type():
    # End-to-end: a PNG screenshot in the trajectory must surface as an image/png frame.
    raw = json.dumps([_shot_span(_PNG_1x1)])
    text, frames = compact_screenshot_trajectory(raw, last_n=1)
    assert frames == [ImageFrame("image/png", _PNG_1x1)]
    assert _PNG_1x1 not in text
    assert "iVBORw0KGgo" not in text


def test_strip_base64_images_removes_all_payload_forms():
    jpeg = "/9j/" + "A" * 500
    png = "iVBORw0KGgo" + "B" * 500
    data_uri = "data:image/png;base64,iVBORw0KGgo" + "C" * 500
    text = f'pre {jpeg} mid {png} and "img":"{data_uri}" post'
    stripped = strip_base64_images(text)
    assert "/9j/" not in stripped
    assert "iVBORw0KGgo" not in stripped
    assert "data:image" not in stripped
    assert "pre " in stripped and " post" in stripped  # surrounding text preserved
    assert stripped.count("<image omitted") == 3


def test_strip_base64_images_leaves_short_tokens_alone():
    # A short string that happens to start with the JPEG magic must NOT be nuked.
    text = "value /9j/abc and normal words"
    assert strip_base64_images(text) == text


# --------------------------------------------------------------------------- #
# compact_screenshot_trajectory — the SCREENSHOT compaction variant

def test_compact_screenshot_returns_stripped_text_and_frames():
    big = "/9j/" + "A" * 500
    raw = json.dumps([_shot_span(big), _shot_span("/9j/" + "B" * 500)])
    text, frames = compact_screenshot_trajectory(raw, last_n=1)
    assert frames == [ImageFrame("image/jpeg", "/9j/" + "B" * 500)]  # neutral frames, last-N
    assert "/9j/" not in text                                         # all base64 stripped
    assert "<image omitted" in text


def test_compact_screenshot_truncates_keeping_tail():
    raw = "HEAD_START_" + ("x" * 200) + "_TAIL_END"   # not JSON → no frames
    text, frames = compact_screenshot_trajectory(raw, last_n=3, max_chars=40)
    assert frames == []
    assert "_TAIL_END" in text and "HEAD_START_" not in text
    assert "showing the final portion" in text


def test_compact_screenshot_no_screenshots_no_blocks():
    raw = json.dumps([{"attributes": {"gen_ai.completion": json.dumps({"result": "ok"})}}])
    text, frames = compact_screenshot_trajectory(raw, last_n=3)
    assert frames == []


def test_compact_screenshot_placeholder_honest_when_no_frames():
    # Chat/prompt-history image but NO completion.screenshot → images stripped, none
    # attached. The placeholder must not claim frames were attached.
    chat_png = "iVBORw0KGgo" + "B" * 500
    raw = json.dumps([{"attributes": {"gen_ai.completion": json.dumps(
        {"content": [{"type": "image_url", "image_url": {"url": "x"}}, {"history": chat_png}]})}}])
    text, frames = compact_screenshot_trajectory(raw, last_n=3)
    assert frames == []
    assert chat_png not in text                       # the history image was stripped
    assert "attached separately" not in text          # ...but we don't claim it was attached
    assert "<image omitted>" in text                  # neutral placeholder


def test_compact_screenshot_placeholder_claims_attachment_only_with_frames():
    raw = json.dumps([_shot_span("/9j/" + "A" * 500)])
    text, frames = compact_screenshot_trajectory(raw, last_n=1)
    assert frames and "attached separately" in text


def test_screenshot_filter_rejects_non_positive_last_n():
    import pytest
    with pytest.raises(ValueError):
        TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT, screenshot_last_n=0)
    with pytest.raises(ValueError):
        TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT, screenshot_last_n=999)
    # A DEFAULT filter doesn't use screenshot_last_n, so it isn't validated.
    TrajectoryFilter(compaction_type=CompactionType.DEFAULT, screenshot_last_n=0)


def test_verifier_rejects_api_base_with_agent_judge():
    import pytest
    with pytest.raises(ValueError, match="direct LLM judge"):
        RubricsVerifierTaskStep(
            id="v", version=1, criteria=[], prompt_id="p",
            use_agent_judge=True, default_model_api_base="https://proxy.example/v1",
        )


def test_schema_fallback_only_triggers_on_response_format_errors():
    # Needs BOTH a schema/response-format term AND an unsupported term → fall back.
    assert _looks_like_response_format_error(Exception("response_format is not supported"))
    assert _looks_like_response_format_error(Exception("json_schema not allowed with images"))
    assert _looks_like_response_format_error(Exception("structured output unsupported here"))
    # Unsupported-IMAGE errors are NOT schema problems (no schema term) → keep schema.
    assert not _looks_like_response_format_error(Exception("unsupported image type: image/webp"))
    assert not _looks_like_response_format_error(Exception("model does not support image_url"))
    assert not _looks_like_response_format_error(Exception("image size unsupported"))
    # A bare schema mention with no incompatibility term → not conclusive → keep schema.
    assert not _looks_like_response_format_error(Exception("response_format must be an object"))
    # Transient blips → do NOT fall back; retry with the schema intact.
    assert not _looks_like_response_format_error(Exception("Connection reset by peer"))
    assert not _looks_like_response_format_error(Exception("503 Service Unavailable"))
    assert not _looks_like_response_format_error(TimeoutError("request timed out"))


# --------------------------------------------------------------------------- #
# Original bug: frames must come from the RAW trajectory, not the compacted one.

def test_compaction_strips_screenshots_so_raw_is_required():
    # A screenshot big enough that compaction externalizes it to a side file.
    big_b64 = "A" * 4096
    raw_spans = [_shot_span(big_b64)]
    raw_text = json.dumps(raw_spans)

    # RAW trajectory → frame is found.
    assert _frames(raw_text, last_n=3) == [big_b64]

    # Run the REAL compaction path the verifier uses for the judge's text.
    events, externalized = compact_otel_trajectory(raw_spans, TrajectoryFilter())
    compacted_text = json.dumps(events)

    # The screenshot was externalized out of the compacted trajectory...
    assert externalized, "expected the screenshot to be externalized by compaction"
    assert big_b64 not in compacted_text
    # ...so extracting from the COMPACTED text finds nothing (the original bug).
    assert _frames(compacted_text, last_n=3) == []


# --------------------------------------------------------------------------- #
# execute() wiring: SCREENSHOT compaction → frames attached + base64-free text.

@pytest.mark.asyncio
async def test_execute_screenshot_compaction_attaches_frames_and_strips_text(monkeypatch):
    # Base64 lives in BOTH a chat/prompt-history span and a tool-result screenshot.
    # The SCREENSHOT compaction must attach the final frame AND strip all base64
    # from the judge's text — and must NOT trigger the default S3 compaction.
    jpeg = "/9j/" + "A" * 500          # execute_tool screenshot → the frame source
    chat_data_uri = "data:image/jpeg;base64,/9j/" + "C" * 500  # prompt-history image
    chat_png = "iVBORw0KGgo" + "B" * 500
    raw_spans = [
        {"attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.completion": json.dumps({"content": [
                {"type": "text", "text": "looking at the screen"},
                {"type": "image_url", "image_url": {"url": chat_data_uri}},
            ]}),
            "gen_ai.prompt": json.dumps({"history_image": chat_png}),
        }},
        _shot_span(jpeg),
    ]
    raw_text = json.dumps(raw_spans)
    raw_uri = "s3://bucket/raw.json"

    verifier = RubricsVerifierTaskStep(
        id="verify", version=1,
        criteria=[{"id": "c1", "description": "outcome", "backend": "final_state"}],
        prompt_id="prompt-1",
        use_agent_judge=False, use_trajectory=True,
        trajectory_filter=_screenshot_filter(3),
        verifier_id="verifier-test",
    )

    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda uri: raw_text)

    def _no_default_compaction(*a, **k):
        raise AssertionError("default _filter_trajectory must NOT run for screenshot compaction")

    monkeypatch.setattr(verifier, "_filter_trajectory", _no_default_compaction)

    captured: dict = {}

    async def fake_run(*, eval_prompt, context, model, image_blocks=None, **kwargs):
        captured["eval_prompt"] = eval_prompt
        captured["image_blocks"] = image_blocks
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)

    ctx = TaskStepContext(prompt_responses=[
        PromptResponse(prompt_id="prompt-1", response="done", prompt_text="do it",
                       agent_trajectory_s3_uri=raw_uri),
    ])

    await verifier.execute(ctx)

    text = captured["eval_prompt"]
    assert "/9j/" not in text and "iVBORw0KGgo" not in text and "data:image" not in text
    assert len(text) < 5000, "stripped prompt should be small, not multi-MB"
    assert "Final-state screenshots" in text                    # grounding appended
    assert captured["image_blocks"] == [_expected_block(jpeg)]  # final frame attached from raw
    assert ctx.metadata["verifications"]["verifier-test"]["score"] == 1.0


@pytest.mark.asyncio
async def test_execute_truncation_keeps_the_tail(monkeypatch):
    # A trajectory longer than the cap: the END (final_state proof) must survive,
    # the early navigation gets dropped.
    raw_text = "HEAD_START_" + ("x" * 200) + "_TAIL_END"
    raw_uri = "s3://bucket/raw.json"

    verifier = RubricsVerifierTaskStep(
        id="verify", version=1,
        criteria=[{"id": "c1", "description": "outcome", "backend": "final_state"}],
        prompt_id="prompt-1",
        use_agent_judge=False, use_trajectory=True,
        trajectory_filter=_screenshot_filter(3),
        verifier_id="verifier-test",
    )
    monkeypatch.setattr("agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter.MAX_JUDGE_TRAJECTORY_CHARS", 40)
    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda uri: raw_text)

    captured: dict = {}

    async def fake_run(*, eval_prompt, context, model, image_blocks=None, **kwargs):
        captured["eval_prompt"] = eval_prompt
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)

    ctx = TaskStepContext(prompt_responses=[
        PromptResponse(prompt_id="prompt-1", response="done", prompt_text="do it",
                       agent_trajectory_s3_uri=raw_uri),
    ])
    await verifier.execute(ctx)

    text = captured["eval_prompt"]
    assert "_TAIL_END" in text          # ending preserved
    assert "HEAD_START_" not in text     # early portion dropped
    assert "showing the final portion" in text


@pytest.mark.asyncio
async def test_execute_honors_apply_trajectory_filter_false(monkeypatch):
    # apply_trajectory_filter=False must DISABLE screenshot compaction entirely:
    # no frames attached, no grounding, the judge runs the plain (unfiltered) text path.
    jpeg = "/9j/" + "A" * 500
    raw_text = json.dumps([_shot_span(jpeg)])
    raw_uri = "s3://bucket/raw.json"

    verifier = RubricsVerifierTaskStep(
        id="verify", version=1,
        criteria=[{"id": "c1", "description": "outcome", "backend": "final_state"}],
        prompt_id="prompt-1",
        use_agent_judge=False, use_trajectory=True,
        trajectory_filter=_screenshot_filter(3),
        verifier_id="verifier-test",
    )
    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda uri: raw_text)

    captured: dict = {}

    async def fake_run(*, eval_prompt, context, model, image_blocks=None, **kwargs):
        captured["eval_prompt"] = eval_prompt
        captured["image_blocks"] = image_blocks
        return ([{**verifier.criteria[0], "score": 1.0, "result": True}], 0, [], None)

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)

    ctx = TaskStepContext(
        prompt_responses=[PromptResponse(prompt_id="prompt-1", response="done",
                                         prompt_text="do it", agent_trajectory_s3_uri=raw_uri)],
        metadata={"user_overrides": {"apply_trajectory_filter": False}},
    )
    await verifier.execute(ctx)

    # Override honored: no frames, no grounding, and the trajectory is unfiltered
    # (raw base64 passes straight through — the explicit "don't filter" choice).
    assert captured["image_blocks"] == []
    assert "Final-state screenshots" not in captured["eval_prompt"]
    assert jpeg in captured["eval_prompt"]


@pytest.mark.asyncio
async def test_prompt_llm_judge_base_url_precedence(monkeypatch):
    # No provider inference in the verifier: the model string is passed through verbatim
    # (litellm resolves the provider from it), and api_base follows the precedence
    # override → default_model_api_base → env.
    import types

    import litellm
    import agent_env.config as cfg_mod
    import agent_env.utils.litellm_attribution as attr_mod

    from agent_env.store import LocalSecretStore

    real_cfg = cfg_mod.Config()
    real_cfg.set_secret_store(LocalSecretStore(values={"litellm_api_key": "k"}, use_env=False))
    monkeypatch.setattr(real_cfg, "_model_cfg", ModelConfig())
    monkeypatch.setattr(cfg_mod, "get_config", lambda: real_cfg)
    monkeypatch.setattr(attr_mod, "build_litellm_cost_attribution_kwargs", lambda md: {})
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        msg = types.SimpleNamespace(content='{"results": []}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    v = RubricsVerifierTaskStep(
        id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=False,
        default_model_api_base="https://verifier-base.example/v1", verifier_id="vt",
    )

    # Model passed through verbatim; no inferred provider; base = default_model_api_base.
    await v._prompt_llm_judge("eval", model="openai/gemini-pro-latest",
                              context=TaskStepContext(), image_blocks=None)
    assert calls[-1]["model"] == "openai/gemini-pro-latest"
    assert "custom_llm_provider" not in calls[-1]
    assert calls[-1]["api_base"] == "https://verifier-base.example/v1"

    # A per-run override wins the base precedence.
    ctx_override = TaskStepContext(
        metadata={"user_overrides": {"litellm_base_url": "https://override.example/v1"}})
    await v._prompt_llm_judge("eval", model="openai/gemini-pro-latest",
                              context=ctx_override, image_blocks=None)
    assert calls[-1]["api_base"] == "https://override.example/v1"

    # No field / no override → env-derived base (via LITELLM_BASE_URL → resolve_model_call).
    monkeypatch.setenv("LITELLM_BASE_URL", "https://env-proxy.example")
    v_env = RubricsVerifierTaskStep(
        id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=False, verifier_id="vt",
    )
    await v_env._prompt_llm_judge("eval", model="claude-sonnet-4-6",
                                  context=TaskStepContext(), image_blocks=None)
    assert calls[-1]["api_base"] == "https://env-proxy.example"
