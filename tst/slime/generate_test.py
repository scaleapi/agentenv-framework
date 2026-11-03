"""Rows ``write_rows`` wrote, played by ``agent_env.training.slime.generate`` as slime's rollout calls it: with slime's
own arguments, through slime's own OpenAI adapter in front of a stand-in for SGLang's ``/generate``, which plays the
model from a script with Qwen3's tokenizer; slime's dynamic-sampling filter decides each group."""

import asyncio
import json
import threading
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_env.training._chat import write_rows
from agent_env.training._openenv import Server
from tst.unit.training._support import ADDS_A_CRATE, FAILS_AFTER_THE_TURN, PLAYED
from tst.util.capabilities import SLIME, missing_capability_reason

slime_generate = pytest.importorskip("agent_env.training.slime.generate", reason=missing_capability_reason(SLIME),
                                     exc_type=ImportError)
arguments = pytest.importorskip("slime.utils.arguments", reason=missing_capability_reason(SLIME))
filters = pytest.importorskip("slime.rollout.filter_hub.base_types", reason=missing_capability_reason(SLIME))
types = pytest.importorskip("slime.utils.types", reason=missing_capability_reason(SLIME))
transformers = pytest.importorskip("transformers", reason=missing_capability_reason(SLIME))

CHECKPOINT = "Qwen/Qwen3-0.6B"
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 256}


@pytest.fixture(autouse=True)
def fresh_adapter(monkeypatch):
    """Each test's own adapter, in front of its own stand-in SGLang: the module keeps one per process."""
    monkeypatch.setattr(slime_generate._Adapter, "served", None)


@pytest.fixture
def model():
    """A stand-in for SGLang's ``/generate``: adds a crate, then says so once the tool has answered, that reply held
    back ``.stall`` seconds with ``.stalled`` set meanwhile; or answers 500 when ``.broken``."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)

    async def generate(request):
        if server.broken:
            return JSONResponse({"error": "engine down"}, status_code=500)
        prompt = tokenizer.decode((await request.json())["input_ids"])
        if "<tool_response>" in prompt and server.stall:
            server.stalled.set()
            await asyncio.sleep(server.stall)
        ids = tokenizer.encode(f"{'Added a crate.' if '<tool_response>' in prompt else ADDS_A_CRATE}<|im_end|>",
                               add_special_tokens=False)
        return JSONResponse({"meta_info": {"output_token_logprobs": [[-0.01, each] for each in ids],
                                           "finish_reason": {"type": "stop"}}})

    server = Server(Starlette(routes=[Route("/generate", generate, methods=["POST"])]))
    server.broken, server.stall, server.stalled, server.tokenizer = False, 0.0, threading.Event(), tokenizer
    yield server
    server.stop()


def test_slime_plays_a_row_through_its_adapter_and_trains_on_the_models_turns_with_the_episodes_reward(
        serve, model, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)

    samples = _generate(row, _args(model, tmp_path))
    server.stop()

    assert {sample.reward for sample in samples} == {1.0}
    assert all(sample.metadata["agentenv_scores"] == {"replies": 1.0} and sample.metadata["agentenv_tool_calls"] == 1
               for sample in samples)
    trained = "".join(_trained(sample, model.tokenizer) for sample in samples)
    assert '"crate"' in trained and "Added a crate." in trained and "<tool_response>" not in trained


def test_an_eval_gives_slime_one_sample_per_episode_so_its_mean_is_over_episodes(serve, model, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)
    args = _args(model, tmp_path)

    [sample] = _generate(row, args, evaluation=True)
    server.stop()

    assert (sample.reward, sample.status, sample.loss_mask) == (1.0, types.Sample.Status.COMPLETED, [0])
    assert sample.metadata["agentenv_scores"] == {"replies": 1.0}
    assert slime_generate._Adapter.of(args).adapter.store == {}


def test_a_rollout_slime_cancels_drops_its_session_and_closes_its_episode(serve, model, tmp_path, teardowns):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)
    args = _args(model, tmp_path)
    model.stall = 5.0

    async def cancelled():
        rollout = asyncio.ensure_future(slime_generate.generate(args, _sample(row), dict(SAMPLING)))
        while not model.stalled.is_set():
            await asyncio.sleep(0.05)
        rollout.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rollout

    asyncio.run(cancelled())
    server.stop()

    assert slime_generate.generate.abort_mode == "request"
    assert slime_generate._Adapter.of(args).adapter.store == {}
    assert len(teardowns.finished) == 2


def test_an_episode_the_env_failed_is_flagged_and_slime_leaves_its_group_out(serve, model, tmp_path):
    server = serve({"a": FAILS_AFTER_THE_TURN})
    row = _row(server, tmp_path)
    args = _args(model, tmp_path)

    samples = _generate(row, args)
    server.stop()

    assert samples and all(sample.metadata["agentenv_failure"] for sample in samples)
    assert filters.call_dynamic_filter(slime_generate.keep_scored_groups, args, samples).keep is False


def test_an_episode_with_no_recorded_turn_comes_back_as_one_aborted_sample_slime_leaves_out(serve, model, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)
    args = _args(model, tmp_path)
    model.broken = True

    [sample] = _generate(row, args)
    server.stop()

    assert (sample.status, sample.remove_sample, sample.reward) == (types.Sample.Status.ABORTED, True, 0.0)
    assert sample.metadata["agentenv_failure"]
    assert filters.call_dynamic_filter(slime_generate.keep_scored_groups, args, [sample]).keep is False


def test_the_adapter_needs_slimes_tool_call_parser(serve, model, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)
    args = _args(model, tmp_path)
    args.sglang_tool_call_parser = None

    with pytest.raises(ValueError, match="--sglang-tool-call-parser"):
        _generate(row, args)
    server.stop()


def _row(server, tmp_path):
    folder = write_rows(server.url, tmp_path / "rows")
    return json.loads((folder / "all.jsonl").read_text().splitlines()[0])


def _generate(row, args, evaluation=False):
    """The samples slime's rollout gets back for ``row``, as it calls the generate function."""
    return asyncio.run(slime_generate.generate(args, _sample(row), dict(SAMPLING), evaluation=evaluation))


def _sample(row):
    return types.Sample(prompt=row["messages"], metadata=row["metadata"], index=0)


def _trained(sample, tokenizer):
    """The text of the tokens slime trains on in ``sample``."""
    response = sample.tokens[len(sample.tokens) - sample.response_length:]
    return tokenizer.decode([token for token, trained in zip(response, sample.loss_mask) if trained])


def _args(model, tmp_path):
    """slime's own arguments for a rollout of the rows with the helper, in front of the stand-in ``model``; a two-layer
    model shape keeps Megatron's checks satisfied."""
    port = int(model.url.rsplit(":", 1)[1])
    argv = [
        "train.py", "--hf-checkpoint", CHECKPOINT, "--rollout-num-gpus", "1", "--rollout-num-gpus-per-engine", "1",
        "--num-layers", "2", "--hidden-size", "64", "--num-attention-heads", "4", "--ffn-hidden-size", "128",
        "--seq-length", "512", "--max-position-embeddings", "512", "--debug-rollout-only",
        "--prompt-data", str(tmp_path / "rows" / "all.jsonl"), "--input-key", "messages", "--metadata-key", "metadata",
        "--rollout-batch-size", "1", "--n-samples-per-prompt", "1", "--global-batch-size", "1", "--num-rollout", "1",
        "--rollout-max-context-len", "8192", "--apply-chat-template-kwargs", json.dumps({"enable_thinking": False}),
        "--sglang-router-ip", "127.0.0.1", "--sglang-router-port", str(port), "--sglang-tool-call-parser", "qwen25",
        "--custom-generate-function-path", "agent_env.training.slime.generate.generate",
        "--dynamic-sampling-filter-path", "agent_env.training.slime.generate.keep_scored_groups",
    ]
    with patch("sys.argv", argv):
        return arguments.parse_args()
