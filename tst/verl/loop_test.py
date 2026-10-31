"""``AgentEnvAgentLoop`` on verl itself: verl's ``ToolAgentLoop`` plays a row ``write_dataset`` wrote, with Qwen3's
tokenizer and chat template and a stand-in rollout server, on the unit tier's bundle served on loopback. A played row's
reward is its episode's, through verl's hermes and qwen3_coder parsers; a rollout the env fails is played again on a new
episode, but not once its time is up, and one it fails every time is kept out of the loss; a prompt that doesn't fit and
an episode that ends unscored without failing raise. CI runs it on verl 0.9.1 and main."""

import asyncio
import json

import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from agent_env.training._openenv import Failed
from agent_env.training.verl.prepare import write_dataset
from tst.unit.training.nemo_gym_test import PLAYED
from tst.util.capabilities import VERL, missing_capability_reason

verl_loop = pytest.importorskip("agent_env.training.verl.loop", reason=missing_capability_reason(VERL),
                                exc_type=ImportError)
omegaconf = pytest.importorskip("omegaconf", reason=missing_capability_reason(VERL))
agent_loop = pytest.importorskip("verl.experimental.agent_loop.agent_loop", reason=missing_capability_reason(VERL))
tool_parser = pytest.importorskip("verl.experimental.agent_loop.tool_parser", reason=missing_capability_reason(VERL))
rl_dataset = pytest.importorskip("verl.utils.dataset.rl_dataset", reason=missing_capability_reason(VERL))
replica = pytest.importorskip("verl.workers.rollout.replica", reason=missing_capability_reason(VERL))

TOKENIZER = "Qwen/Qwen3-0.6B"
DATA = omegaconf.OmegaConf.create({"apply_chat_template_kwargs": {"enable_thinking": False}})
ADD_A_CRATE = {
    "hermes": '<tool_call>\n{"name": "add_item", "arguments": {"item": "crate"}}\n</tool_call><|im_end|>',
    "qwen3_coder": ("<tool_call>\n<function=add_item>\n<parameter=item>\ncrate\n</parameter>\n</function>\n"
                    "</tool_call><|im_end|>"),
}
SCHEDULE = {"type": "function", "function": {"name": "submit_schedule", "description": "Submit the day's schedule.",
                                              "parameters": {"type": "object", "required": ["schedule"], "properties": {
                                                  "schedule": {"type": "array", "items": {"type": "integer"},
                                                               "description": "One action an hour."},
                                                  "hour": {"type": "integer"},
                                                  "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.mark.parametrize("parser", ["hermes", "qwen3_coder"])
def test_a_played_rows_reward_is_its_episodes_and_its_last_reply_ends_the_turn(serve, tmp_path, tokenizer, parser,
                                                                                 monkeypatch, teardowns):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    model = _Model(tokenizer, parser)
    finish, replies = verl_loop._Session._finish, []

    async def finishing(self, rollout, response=""):
        replies.append(response)
        await finish(self, rollout, response)

    monkeypatch.setattr(verl_loop._Session, "_finish", finishing)
    output = asyncio.run(_play(server.url, row, model, tokenizer, parser=parser))
    server.stop()

    assert (output.reward_score, output.extra_fields["agentenv"]) == (1.0, {"attempts": 1, "scores": {"replies": 1.0}})
    assert (model.calls, replies) == (2, ["Added a crate."])
    assert 0 in output.response_mask and 1 in output.response_mask
    assert "Added a crate." in tokenizer.decode(output.response_ids)
    assert len(teardowns.finished) == 2


def test_a_tools_parameters_reach_verls_parsers_with_their_types(tokenizer):
    schema = verl_loop._parser_schema(SCHEDULE)
    call = ("<tool_call>\n<function=submit_schedule>\n<parameter=schedule>\n[1, 0, -1]\n</parameter>\n"
            "<parameter=hour>\n7\n</parameter>\n<parameter=note>\nnone today\n</parameter>\n</function>\n</tool_call>")
    parser = tool_parser.ToolParser.get_tool_parser("qwen3_coder", tokenizer)

    _, [parsed] = asyncio.run(parser.extract_tool_calls(tokenizer.encode(call, add_special_tokens=False), [schema]))

    assert (schema.function.name, schema.function.parameters.required) == ("submit_schedule", ["schedule"])
    assert {name: spec.model_dump(exclude_none=True) for name, spec in schema.function.parameters.properties.items()} == {
        "schedule": {"type": "array", "description": "One action an hour."}, "hour": {"type": "integer"}}
    assert (parsed.name, json.loads(parsed.arguments)) == (
        "submit_schedule", {"schedule": [1, 0, -1], "hour": 7, "note": "none today"})


def test_a_rollout_the_env_fails_is_played_again_on_a_new_episode(serve, tmp_path, tokenizer, monkeypatch, teardowns):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    begin, began = verl_loop._Session._begin, []

    async def failing_once(self, rollout, *args):
        began.append(1)
        if len(began) == 1:
            raise Failed("the env's deploy failed")
        return await begin(self, rollout, *args)

    monkeypatch.setattr(verl_loop._Session, "_begin", failing_once)
    output = asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer))
    server.stop()

    assert (output.reward_score, output.extra_fields["agentenv"]["attempts"]) == (1.0, 2)
    assert len(began) == 2


def test_a_rollout_the_env_fails_every_time_is_kept_out_of_the_loss(serve, tmp_path, tokenizer, monkeypatch):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))

    async def failing(self, rollout, *args):
        raise Failed("the env's deploy failed")

    monkeypatch.setattr(verl_loop._Session, "_begin", failing)
    output = asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, attempts=2))
    server.stop()

    assert (output.reward_score, output.extra_fields["agentenv"]) == (
        0.0, {"attempts": 2, "failure": "the env's deploy failed"})
    assert output.response_mask and set(output.response_mask) == {0}


def test_a_rollout_isnt_played_again_once_its_time_is_up(serve, tmp_path, tokenizer, monkeypatch):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))

    async def failing_late(self, rollout, *args):
        await asyncio.sleep(0.3)
        raise Failed("the env's deploy missed its deadline")

    monkeypatch.setattr(verl_loop._Session, "_begin", failing_late)
    output = asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, reset_timeout=0.2))
    server.stop()

    assert (output.reward_score, output.extra_fields["agentenv"]) == (
        0.0, {"attempts": 1, "failure": "the env's deploy missed its deadline"})


def test_a_prompt_that_doesnt_fit_with_its_tools_is_refused_rather_than_cut(serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))

    with pytest.raises(ValueError, match=r"task a of split all's prompt is \d+ tokens with its tools, over "
                                         r"rollout\.prompt_length=64"):
        asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, prompt_length=64))
    server.stop()


def test_an_episode_that_ends_without_failing_or_a_reward_raises(serve, tmp_path, tokenizer, monkeypatch):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    model = _Model(tokenizer)
    finish = verl_loop._Session._finish

    async def unscored(self, rollout, response=""):
        await finish(self, rollout, response)
        rollout.reward = None

    monkeypatch.setattr(verl_loop._Session, "_finish", unscored)
    with pytest.raises(ValueError, match="task a of split all ended without a reward: it records no score"):
        asyncio.run(_play(server.url, row, model, tokenizer))
    server.stop()

    assert model.calls == 2


class _Model:
    """verl's rollout server for the loop: it adds a crate in ``parser``'s format, then replies once the tool
    answered."""

    def __init__(self, tokenizer, parser="hermes"):
        self.tokenizer, self.parser, self.calls = tokenizer, parser, 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
        self.calls += 1
        answered = "<tool_response>" in self.tokenizer.decode(prompt_ids)
        reply = "Added a crate.<|im_end|>" if answered else ADD_A_CRATE[self.parser]
        return replica.TokenOutput(token_ids=self.tokenizer.encode(reply, add_special_tokens=False))


async def _play(url, row, model, tokenizer, *, parser="hermes", prompt_length=2048, reset_timeout=60.0, attempts=3):
    config = omegaconf.OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "prompt_length": prompt_length, "response_length": 1024, "full_determinism": False,
        "multi_turn": {"format": parser, "max_user_turns": None, "max_assistant_turns": 4, "max_parallel_calls": 1,
                       "max_tool_response_length": 2048, "tool_response_truncate_side": "middle"},
    }}})
    played = verl_loop.AgentEnvAgentLoop(
        trainer_config=agent_loop.DictConfigWrap(config), server_manager=model, tokenizer=tokenizer, processor=None,
        dataset_cls=rl_dataset.RLHFDataset, data_config=agent_loop.DictConfigWrap(DATA), hf_model_type="qwen3",
        url=url, reset_timeout=reset_timeout, attempts=attempts)
    return await played.run({"temperature": 1.0}, raw_prompt=row["prompt"], extra_info=row["extra_info"],
                            tools_kwargs={}, index=0, agent_name="agentenv")


def _row(folder):
    [row] = pq.read_table(folder / "all.parquet").to_pylist()
    assert json.loads(row["extra_info"]["tools"])[0]["function"]["name"] == "add_item"
    return row
