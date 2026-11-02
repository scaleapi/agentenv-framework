"""``AgentEnvAgentLoop`` on verl itself: verl's ``ToolAgentLoop`` plays a row ``write_dataset`` wrote, with Qwen3's
tokenizer and chat template and a stand-in rollout server, on the unit tier's bundle served on loopback. A played row's
reward is its episode's, and its scores, attempts and seconds its reward_extra_info, through verl's hermes and
qwen3_coder parsers and from a server that asks for a token; a rollout the env fails is played again on a new episode,
but not once its time is up, and one it fails every time raises, which verl's V1 trainer leaves out; a prompt that
doesn't fit, an episode that ends unscored without failing and a loop run twice raise; an answer verl cuts is warned of;
and the writer counts prompts with their tools in the model's tokens. CI runs it on verl 0.9.1 and main."""

import asyncio
import json
import logging
import re

import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from agent_env.training.verl.prepare import LOOP, write_dataset
from tst.unit.task.episode import run_test
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
    finish, replies = verl_loop._Session.finish, []

    async def finishing(self, reply):
        replies.append(reply)
        return await finish(self, reply)

    monkeypatch.setattr(verl_loop._Session, "finish", finishing)
    output = asyncio.run(_play(server.url, row, model, tokenizer, parser=parser))
    server.stop()

    info = output.extra_fields["reward_extra_info"]
    assert (output.reward_score, info["attempts"], info["score/replies"]) == (1.0, 1, 1.0)
    assert set(info) == {"attempts", "model_seconds", "env_seconds", "score/replies"}
    assert info["env_seconds"] >= output.metrics.compute_score > 0 and info["model_seconds"] >= 0
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


def test_a_rollout_the_env_fails_is_played_again_on_a_new_episode(serve, tmp_path, tokenizer, monkeypatch):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    deploy, deploys = run_test._Env.execute, []

    async def failing_once(self, context):
        deploys.append(self.id)
        if len(deploys) == 1:
            raise RuntimeError("the env didn't come up")
        return await deploy(self, context)

    monkeypatch.setattr(run_test._Env, "execute", failing_once)
    output = asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer))
    server.stop()

    assert (output.reward_score, output.extra_fields["reward_extra_info"]["attempts"], len(deploys)) == (1.0, 2, 2)


def test_a_rollout_the_env_fails_every_time_raises_so_verl_leaves_it_out(serve, tmp_path, tokenizer):
    healthy = serve({"a": PLAYED})
    row = _row(write_dataset(healthy.url, tmp_path))
    healthy.stop()
    broken = serve({"a": [{**PLAYED[0], "fail": True}, *PLAYED[1:]]})

    with pytest.raises(verl_loop.RolloutFailed, match=r"task a of split all: the env failed it on 2 attempts, so it's "
                                                       r"left out of training: .*the env didn't come up"):
        asyncio.run(_play(broken.url, row, _Model(tokenizer), tokenizer, attempts=2))
    broken.stop()


def test_a_rollout_isnt_played_again_once_its_time_is_up(serve, tmp_path, tokenizer, monkeypatch):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    deploy = run_test._Env.execute

    async def hanging(self, context):
        await asyncio.sleep(2)
        return await deploy(self, context)

    monkeypatch.setattr(run_test._Env, "execute", hanging)
    with pytest.raises(verl_loop.RolloutFailed, match="the env failed it on 1 attempt, so"):
        asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, reset_timeout=0.5))
    server.stop()


def test_a_prompt_that_doesnt_fit_with_its_tools_is_refused_rather_than_cut(serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))

    with pytest.raises(ValueError, match=r"task a of split all's prompt is \d+ tokens with its tools, over "
                                         r"rollout\.prompt_length=64"):
        asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, prompt_length=64))
    server.stop()


def test_an_episode_that_ends_without_failing_or_a_reward_raises(serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED}, reward=lambda scores: None)
    row = _row(write_dataset(server.url, tmp_path))
    model = _Model(tokenizer)

    with pytest.raises(ValueError, match="task a of split all ended without a reward: it records no score"):
        asyncio.run(_play(server.url, row, model, tokenizer))
    server.stop()

    assert model.calls == 2


def test_an_agent_loop_plays_one_rollout(serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))

    async def twice():
        played = _loop(server.url, _Model(tokenizer), tokenizer)
        assert (await _run(played, row)).reward_score == 1.0
        await _run(played, row)

    with pytest.raises(RuntimeError, match="verl ran an AgentEnvAgentLoop a second time"):
        asyncio.run(twice())
    server.stop()


def test_a_server_that_asks_for_a_token_is_played_with_the_one_its_variable_holds(serve, tmp_path, tokenizer,
                                                                                 monkeypatch):
    server = serve({"a": PLAYED}, token="s3cret")
    monkeypatch.setenv("DOCK_TOKEN", "s3cret")
    row = _row(write_dataset(server.url, tmp_path, token_var="DOCK_TOKEN"))

    played = asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, token_var="DOCK_TOKEN"))
    with pytest.raises(verl_loop.RolloutFailed, match="couldn't open a session on the server"):
        asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, attempts=1))
    server.stop()

    assert played.reward_score == 1.0


def test_an_answer_verl_cuts_is_warned_of_once_a_tool(serve, tmp_path, tokenizer, monkeypatch, caplog):
    server = serve({"a": PLAYED})
    row = _row(write_dataset(server.url, tmp_path))
    monkeypatch.setattr(verl_loop, "_cut", set())

    with caplog.at_level(logging.WARNING, logger=verl_loop.__name__):
        for _ in range(2):
            asyncio.run(_play(server.url, row, _Model(tokenizer), tokenizer, max_tool_response_length=4))
    server.stop()

    [warning] = [record.getMessage() for record in caplog.records if "verl cuts" in record.getMessage()]
    assert re.fullmatch(r"add_item answered \d+ characters, which verl cuts to actor_rollout_ref\.rollout\.multi_turn\."
                        r"max_tool_response_length=4, so the model doesn't see all of it: raise it to cover the env's "
                        r"answers", warning)


def test_the_writer_counts_each_prompt_with_its_tools_as_verl_renders_it_and_refuses_one_over_the_limit(
        serve, tmp_path, tokenizer):
    server = serve({"a": PLAYED})

    written = write_dataset(server.url, tmp_path / "data", tokenizer=TOKENIZER)
    thinking_off = write_dataset(server.url, tmp_path / "off", tokenizer=TOKENIZER,
                                 chat_template_kwargs={"enable_thinking": False})
    row = _row(written)
    with pytest.raises(ValueError, match=r"1 rows' prompts are over max_prompt_length=16 tokens with their tools, the "
                                         r"longest task a of split all at \d+, which verl would cut"):
        write_dataset(server.url, tmp_path / "refused", tokenizer=TOKENIZER, max_prompt_length=16)
    server.stop()

    def rendered(**kwargs):
        return len(tokenizer.apply_chat_template(row["prompt"], tools=json.loads(row["extra_info"]["tools"]),
                                                 add_generation_prompt=True, tokenize=True, return_dict=True,
                                                 **kwargs)["input_ids"])

    assert (written.longest_prompt, thinking_off.longest_prompt) == (rendered(), rendered(enable_thinking=False))
    assert thinking_off.longest_prompt != written.longest_prompt > 16
    assert not (tmp_path / "refused").exists()


def test_the_rows_config_names_this_agent_loop():
    assert LOOP == f"{verl_loop.AgentEnvAgentLoop.__module__}.{verl_loop.AgentEnvAgentLoop.__qualname__}"


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


async def _play(url, row, model, tokenizer, **options):
    return await _run(_loop(url, model, tokenizer, **options), row)


def _loop(url, model, tokenizer, *, parser="hermes", prompt_length=2048, max_tool_response_length=2048,
          reset_timeout=60.0, attempts=3, token_var=None):
    config = omegaconf.OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "prompt_length": prompt_length, "response_length": 1024, "full_determinism": False,
        "multi_turn": {"format": parser, "max_user_turns": None, "max_assistant_turns": 4, "max_parallel_calls": 1,
                       "max_tool_response_length": max_tool_response_length, "tool_response_truncate_side": "middle"},
    }}})
    return verl_loop.AgentEnvAgentLoop(
        trainer_config=agent_loop.DictConfigWrap(config), server_manager=model, tokenizer=tokenizer, processor=None,
        dataset_cls=rl_dataset.RLHFDataset, data_config=agent_loop.DictConfigWrap(DATA), hf_model_type="qwen3",
        url=url, reset_timeout=reset_timeout, attempts=attempts, token_var=token_var)


async def _run(played, row):
    return await played.run({"temperature": 1.0}, raw_prompt=row["prompt"], extra_info=row["extra_info"],
                            tools_kwargs={}, index=0, agent_name="agentenv")


def _row(written):
    [row] = pq.read_table(written.folder / "all.parquet").to_pylist()
    assert json.loads(row["extra_info"]["tools"])[0]["function"]["name"] == "add_item"
    return row
