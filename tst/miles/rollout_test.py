"""Rows ``write_rows`` wrote, played by Miles' own rollout: its agentic generate path hands each sample to
``agent_env.training.miles.agent.run`` with a session on Miles' session server, in front of Miles' router and its mock
SGLang server, which plays the model from a script; Miles scores the samples with ``agent.reward`` and groups them
through ``agent.keep_scored_groups``."""

import json
from unittest.mock import patch

import httpx
import pytest

from agent_env.training.miles.prepare import write_rows
from tst.unit.training._support import ADDS_A_CRATE, FAILS_AFTER_THE_TURN, PLAYED
from tst.util.capabilities import MILES, missing_capability_reason

fixtures = pytest.importorskip("tests.fast.fixtures.rollout_fixtures", reason=missing_capability_reason(MILES))
integration = pytest.importorskip("tests.fast.rollout.inference_rollout.integration.utils",
                                  reason=missing_capability_reason(MILES))
arguments = pytest.importorskip("miles.utils.arguments", reason=missing_capability_reason(MILES))
function_registry = pytest.importorskip("miles.utils.function_registry", reason=missing_capability_reason(MILES))
http_utils = pytest.importorskip("miles.utils.http_utils", reason=missing_capability_reason(MILES))
misc = pytest.importorskip("miles.utils.misc", reason=missing_capability_reason(MILES))
mock_sglang = pytest.importorskip("miles.utils.test_utils.mock_sglang_server", reason=missing_capability_reason(MILES))
sample_types = pytest.importorskip("miles.utils.types", reason=missing_capability_reason(MILES))



@pytest.fixture(params=[[], ["v2"]], ids=["session-server-v1", "session-server-v2"])
def session_server(request):
    """Miles' session server, bare (v1, append-only) or ``v2`` (tree serving), which merges the agent function's
    metadata into the samples itself."""
    return ["--use-session-server", *request.param]


def test_miles_plays_a_row_through_its_session_and_trains_on_the_episodes_reward(serve, tmp_path, session_server):
    server = serve({"a": PLAYED})
    rows = write_rows(server.url, tmp_path / "rows") / "all.jsonl"

    played = _rollout(rows, session_server)
    server.stop()

    [[sample]] = _groups(played)
    assert (sample.status, sample.reward) == (sample_types.Sample.Status.COMPLETED, 1.0)
    assert {key: sample.metadata[key] for key in ("agentenv_scores", "agentenv_tool_calls", "agentenv_invalid_calls")} \
        == {"agentenv_scores": {"replies": 1.0}, "agentenv_tool_calls": 1, "agentenv_invalid_calls": 0}
    assert sample.response.startswith(ADDS_A_CRATE) and sample.response.endswith("Added a crate.")


def test_miles_leaves_out_a_group_the_env_failed_and_draws_rows_until_its_batch_is_full(serve, tmp_path,
                                                                                        session_server):
    server = serve({"a": FAILS_AFTER_THE_TURN, "b": PLAYED})
    rows = write_rows(server.url, tmp_path / "rows") / "all.jsonl"

    played = _rollout(rows, session_server, groups=2)
    server.stop()

    assert [[(sample.metadata["task_id"], sample.reward) for sample in group] for group in _groups(played)] == [
        [("b", 1.0)], [("b", 1.0)]]


def _rollout(rows, session_server, groups=1):
    """A train rollout of ``groups`` groups of one sample from ``rows``, the model answering each chat request with
    ``_model``."""
    misc.SingletonMeta.clear_all_instances()
    try:
        with mock_sglang.with_mock_server(process_fn=_model) as model:
            args = _args(rows, session_server, groups, http_utils.find_available_port(20000))
            with fixtures._with_miles_router(args) as router:
                httpx.post(f"{router.url}/add_worker", params={"url": model.url}, timeout=5.0).raise_for_status()
                with fixtures._with_session_server(args, router.url):
                    data_source = function_registry.load_function(args.data_source_path)(args)
                    return integration.load_and_call_rollout(args, data_source, "train")
    finally:
        misc.SingletonMeta.clear_all_instances()


def _groups(played):
    """The rollout's groups, each a flat list of samples: v2 hands back each generate call's samples as a list."""
    return [[sample for each in group for sample in (each if isinstance(each, list) else [each])]
            for group in played.samples]


def _model(prompt):
    """Adds a crate, then says so once the tool has answered."""
    return mock_sglang.ProcessResult(text="Added a crate." if "<tool_response>" in prompt else ADDS_A_CRATE)


def _args(rows, session_server, groups, router_port):
    """The arguments Miles' own rollout tests start from, with the helper's agent function and hooks in place of a
    label and a reward type."""
    argv = [
        "pytest", "--train-backend", "fsdp", "--ci-test", "--no-enable-sample-ownership-checker",
        "--hf-checkpoint", "Qwen/Qwen3-0.6B", "--rollout-num-gpus", "1", "--rollout-num-gpus-per-engine", "1",
        "--num-rollout", "1", "--rollout-batch-size", str(groups), "--n-samples-per-prompt", "1",
        "--rollout-max-response-len", "256",
        "--prompt-data", str(rows), "--input-key", "messages", "--metadata-key", "metadata",
        "--apply-chat-template-kwargs", json.dumps({"enable_thinking": False}),
        "--use-miles-router", "--sglang-router-ip", "127.0.0.1", "--sglang-router-port", str(router_port),
        *integration.MODULAR_ROLLOUT_BASE_ARGV,
        "--custom-generate-function-path", "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path", "agent_env.training.miles.agent.run",
        *session_server, "--tito-model", "qwen3",
        "--custom-rm-path", "agent_env.training.miles.agent.reward",
        "--dynamic-sampling-filter-path", "agent_env.training.miles.agent.keep_scored_groups",
    ]
    with patch("sys.argv", argv):
        args = arguments.parse_args()
    http_utils.init_http_client(args)
    return args
