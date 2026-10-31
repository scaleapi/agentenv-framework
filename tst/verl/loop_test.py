"""``AgentEnvAgentLoop`` on verl itself: verl's ``ToolAgentLoop`` plays a row ``write_dataset`` wrote, with Qwen3's
tokenizer and chat template and a stand-in rollout server, on the unit tier's bundle served on loopback. A played row's
reward is its episode's, a rollout the env fails is played again on a new episode, and one it fails every time is kept
out of the loss. CI runs it on verl 0.9.1."""

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
rl_dataset = pytest.importorskip("verl.utils.dataset.rl_dataset", reason=missing_capability_reason(VERL))
replica = pytest.importorskip("verl.workers.rollout.replica", reason=missing_capability_reason(VERL))

TOKENIZER = "Qwen/Qwen3-0.6B"
CONFIG = omegaconf.OmegaConf.create({"actor_rollout_ref": {"rollout": {
    "prompt_length": 2048, "response_length": 1024, "full_determinism": False,
    "multi_turn": {"format": "hermes", "max_user_turns": None, "max_assistant_turns": 4, "max_parallel_calls": 1,
                   "max_tool_response_length": 2048, "tool_response_truncate_side": "middle"},
}}})
DATA = omegaconf.OmegaConf.create({"apply_chat_template_kwargs": {"enable_thinking": False}})
ADD_A_CRATE = '<tool_call>\n{"name": "add_item", "arguments": {"item": "crate"}}\n</tool_call><|im_end|>'


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def test_a_played_rows_reward_is_its_episodes_and_its_last_reply_ends_the_turn(serve, tmp_path, tokenizer,
                                                                                 monkeypatch, teardowns):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    model = _Model(tokenizer)
    finish, replies = verl_loop._Session._finish, []

    async def finishing(self, rollout, response=""):
        replies.append(response)
        await finish(self, rollout, response)

    monkeypatch.setattr(verl_loop._Session, "_finish", finishing)
    output = asyncio.run(_play(server.url, row, model, tokenizer))
    server.stop()

    assert (output.reward_score, output.extra_fields["agentenv"]) == (1.0, {"attempts": 1, "scores": {"replies": 1.0}})
    assert (model.calls, replies) == (2, ["Added a crate."])
    assert 0 in output.response_mask and 1 in output.response_mask
    assert "Added a crate." in tokenizer.decode(output.response_ids)
    assert len(teardowns.finished) == 2


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


class _Model:
    """verl's rollout server for the loop: it adds a crate, then replies once the tool answered."""

    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
        self.calls += 1
        answered = "<tool_response>" in self.tokenizer.decode(prompt_ids)
        reply = "Added a crate.<|im_end|>" if answered else ADD_A_CRATE
        return replica.TokenOutput(token_ids=self.tokenizer.encode(reply, add_special_tokens=False))


async def _play(url, row, model, tokenizer, attempts=3):
    played = verl_loop.AgentEnvAgentLoop(
        trainer_config=agent_loop.DictConfigWrap(CONFIG), server_manager=model, tokenizer=tokenizer, processor=None,
        dataset_cls=rl_dataset.RLHFDataset, data_config=agent_loop.DictConfigWrap(DATA), hf_model_type="qwen3",
        url=url, reset_timeout=60.0, attempts=attempts)
    return await played.run({"temperature": 1.0}, raw_prompt=row["prompt"], extra_info=row["extra_info"],
                                tools_kwargs={}, index=0, agent_name="agentenv")


def _row(folder):
    [row] = pq.read_table(folder / "all.parquet").to_pylist()
    assert json.loads(row["extra_info"]["tools"])[0]["function"]["name"] == "add_item"
    return row
