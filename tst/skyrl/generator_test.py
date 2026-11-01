"""``AgentEnvGenerator`` and ``AgentEnvTextEnv`` on SkyRL itself: SkyRL's ``PromptDataset`` loads the rows
``write_dataset`` wrote, with Qwen3's and Qwen3.5's tokenizers and chat templates, and its generator plays them through
a stand-in inference engine, on the unit tier's bundle served on loopback. A played row's reward is its episode's, and
its scores, attempts and tool calls its env metrics, through the hermes and qwen3_coder formats, over two turns, and
from a server that asks for a token; the model sees its tools as its chat template renders them; an unparsed call is
answered with why; the turn cap ends the episode; a rollout the env fails is played again on a new episode, but not
once its time is up, and one it fails every attempt masks its group in training but not in an eval; an episode that
ends unscored without failing raises; and the entrypoint registers the env and reads the rows' token variables. CI runs
it on SkyRL 0.4.0 and main."""

import asyncio
import json

import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from agent_env.training.skyrl.prepare import token_vars, write_dataset
from tst.unit.task.episode import run_test
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN
from tst.unit.training.nemo_gym_test import PLAYED
from tst.util.capabilities import SKYRL, missing_capability_reason

skyrl_env = pytest.importorskip("agent_env.training.skyrl.env", reason=missing_capability_reason(SKYRL),
                                exc_type=ImportError)
skyrl_generator = pytest.importorskip("agent_env.training.skyrl.generator", reason=missing_capability_reason(SKYRL),
                                      exc_type=ImportError)
train = pytest.importorskip("agent_env.training.skyrl.train", reason=missing_capability_reason(SKYRL),
                            exc_type=ImportError)
config = pytest.importorskip("skyrl.train.config", reason=missing_capability_reason(SKYRL))
dataset = pytest.importorskip("skyrl.train.dataset.dataset", reason=missing_capability_reason(SKYRL))
generator_utils = pytest.importorskip("skyrl.train.generators.utils", reason=missing_capability_reason(SKYRL))

TOKENIZERS = {"hermes": "Qwen/Qwen3-0.6B", "qwen3_coder": "Qwen/Qwen3.5-0.8B"}
UNTHINKING = {"enable_thinking": False}
TWO_TURNS = [ITEMS, ALICE, TURN, {**TURN, "id": "turn-2", "prompt_id": "turn-2", "prompt": "Add a barrel."}, REPLIES]


def call(tool_format, name, **arguments):
    """A tool call as ``tool_format`` writes it."""
    if tool_format == "hermes":
        return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'
    parameters = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def adds_a_crate(tool_format):
    """The model of a one-turn task: it adds a crate, then replies once the tool answered."""
    return lambda answered: call(tool_format, "add_item", item="crate") if answered == 0 else "Added a crate."


@pytest.fixture(scope="module", autouse=True)
def registered():
    train.register()


@pytest.fixture(scope="module", params=sorted(TOKENIZERS))
def tool_format(request):
    return request.param


@pytest.fixture(scope="module")
def tokenizer(tool_format):
    return AutoTokenizer.from_pretrained(TOKENIZERS[tool_format])


def test_a_played_rows_reward_is_its_episodes_and_its_last_reply_ends_the_turn(serve, tmp_path, tool_format,
                                                                                 tokenizer, monkeypatch, teardowns):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format)
    engine, replies = _Engine(tokenizer, adds_a_crate(tool_format)), []
    finish = skyrl_env._Session.finish

    async def finishing(self, reply):
        replies.append(reply)
        return await finish(self, reply)

    monkeypatch.setattr(skyrl_env._Session, "finish", finishing)
    output = _generate(rows, engine, tokenizer, samples=2)
    server.stop()

    assert [sum(reward) for reward in output["rewards"]] == [1.0, 1.0]
    assert all(1 in mask and 0 in mask for mask in output["loss_masks"])
    assert output["env_metrics"] == [{"tool_calls": 1, "unparsed_calls": 0, "score/replies": 1.0, "attempts": 1,
                                      "failed": 0}] * 2
    assert output["rollout_metrics"]["environment/score/replies"] == 1.0
    assert (replies, engine.calls) == (["Added a crate."] * 2, 4)
    assert len(teardowns.finished) == 3


def test_the_model_sees_its_tools_as_its_chat_template_renders_them(serve, tmp_path, tool_format, tokenizer):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format)
    engine = _Engine(tokenizer, adds_a_crate(tool_format))

    _generate(rows, engine, tokenizer)
    server.stop()

    [row] = pq.read_table(tmp_path / "all.parquet").to_pylist()
    tools = json.loads(row["episode"]["tools"])
    native = tokenizer.apply_chat_template([{"role": "system", "content": "You keep the items."},
                                            {"role": "user", "content": "Add a crate."}], tools=tools,
                                           add_generation_prompt=True, tokenize=True, return_dict=True, **UNTHINKING)
    assert engine.first_prompt == native["input_ids"]


def test_a_task_of_two_turns_is_played_through_end_turn_and_scored_on_both_replies(serve, tmp_path, tool_format,
                                                                                    tokenizer):
    server = serve({"a": TWO_TURNS})
    rows = _rows(server.url, tmp_path, tool_format)

    def two_turns(answered):
        return [call(tool_format, "add_item", item="crate"), call(tool_format, "end_turn", response="Added a crate."),
                call(tool_format, "add_item", item="barrel"), "Added a barrel."][answered]

    engine = _Engine(tokenizer, two_turns)
    output = _generate(rows, engine, tokenizer, max_turns=6)
    server.stop()

    assert [sum(reward) for reward in output["rewards"]] == [1.0]
    assert output["env_metrics"][0]["tool_calls"] == 3
    assert "Add a barrel." in tokenizer.decode(output["response_ids"][0])


def test_an_unparsed_call_is_answered_with_why_and_counted(serve, tmp_path, tool_format, tokenizer):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format)

    def misspelled(answered):
        return [call(tool_format, "add_items", item="crate"), call(tool_format, "add_item", item="crate"),
                "Added a crate."][answered]

    output = _generate(rows, _Engine(tokenizer, misspelled), tokenizer)
    server.stop()

    assert [sum(reward) for reward in output["rewards"]] == [1.0]
    assert (output["env_metrics"][0]["tool_calls"], output["env_metrics"][0]["unparsed_calls"]) == (1, 1)
    assert "Error: there's no tool 'add_items'; the tools are add_item, end_turn" in tokenizer.decode(
        output["response_ids"][0])


def test_the_turn_cap_ends_the_episode_with_the_last_reply(serve, tmp_path, tool_format, tokenizer):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format)
    engine = _Engine(tokenizer, adds_a_crate(tool_format))

    output = _generate(rows, engine, tokenizer, max_turns=1)
    server.stop()

    assert (engine.calls, [sum(reward) for reward in output["rewards"]]) == (1, [0.0])
    assert output["env_metrics"][0]["tool_calls"] == 1


def test_a_rollout_the_env_fails_is_played_again_on_a_new_episode(serve, tmp_path, tool_format, tokenizer,
                                                                   monkeypatch):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format)
    deploy, deploys = run_test._Env.execute, []

    async def failing_once(self, context):
        deploys.append(self.id)
        if len(deploys) == 1:
            raise RuntimeError("the env didn't come up")
        return await deploy(self, context)

    monkeypatch.setattr(run_test._Env, "execute", failing_once)
    output = _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer)
    server.stop()

    assert [sum(reward) for reward in output["rewards"]] == [1.0]
    assert (output["env_metrics"][0]["attempts"], output["env_metrics"][0]["failed"], len(deploys)) == (2, 0, 2)


@pytest.mark.parametrize("step_wise", [False, True])
@pytest.mark.parametrize(("phase", "kept"), [("train", 0.0), ("eval", 1.0)])
def test_a_rollout_the_env_fails_every_attempt_masks_its_group_in_training_but_not_in_an_eval(
        serve, tmp_path, tool_format, tokenizer, monkeypatch, step_wise, phase, kept):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format, attempts=2)
    deploy, deploys = run_test._Env.execute, []

    async def failing_after_the_first(self, context):
        deploys.append(self.id)
        if len(deploys) > 1:
            raise RuntimeError("the env didn't come up")
        return await deploy(self, context)

    monkeypatch.setattr(run_test._Env, "execute", failing_after_the_first)
    output = _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer, samples=2, phase=phase,
                       step_wise=step_wise)
    server.stop()

    last = [metrics for metrics in output["env_metrics"] if "failed" in metrics]
    assert sorted((metrics["failed"], metrics["attempts"]) for metrics in last) == [(0, 1), (1, 2)]
    played = next(row for row, metrics in enumerate(output["env_metrics"]) if metrics.get("failed") == 0)
    assert sum(output["rewards"][played]) == kept
    assert (1 in output["loss_masks"][played]) == (phase == "eval")
    failed = next(row for row, metrics in enumerate(output["env_metrics"]) if metrics.get("failed") == 1)
    assert (output["loss_masks"][failed], sum(output["rewards"][failed])) == ([0], 0.0)
    assert len(deploys) == 3


def test_more_rollouts_than_env_threads_play_in_turn_on_a_server_at_its_limit(serve, tmp_path, tool_format, tokenizer):
    server = serve({"a": PLAYED}, max_concurrent_envs=2)
    rows = _rows(server.url, tmp_path, tool_format)

    output = _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer, samples=6, env_workers=2,
                       reset_timeout=30.0)
    server.stop()

    assert [sum(reward) for reward in output["rewards"]] == [1.0] * 6
    assert [metrics["attempts"] for metrics in output["env_metrics"]] == [1] * 6


def test_a_rollout_isnt_played_again_once_its_time_is_up(serve, tmp_path, tool_format, tokenizer, monkeypatch):
    server = serve({"a": PLAYED})
    rows = _rows(server.url, tmp_path, tool_format)
    deploy = run_test._Env.execute

    async def hanging(self, context):
        await asyncio.sleep(2)
        return await deploy(self, context)

    monkeypatch.setattr(run_test._Env, "execute", hanging)
    output = _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer, reset_timeout=0.5)
    server.stop()

    assert (output["env_metrics"][0]["failed"], output["env_metrics"][0]["attempts"]) == (1, 1)


def test_an_episode_that_ends_without_failing_or_a_reward_raises(serve, tmp_path, tool_format, tokenizer):
    server = serve({"a": PLAYED}, reward=lambda scores: None)
    rows = _rows(server.url, tmp_path, tool_format)

    with pytest.raises(ValueError, match="task a of split all ended without a reward: it records no score"):
        _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer)
    server.stop()


def test_a_server_that_asks_for_a_token_is_played_with_the_one_its_variable_holds(serve, tmp_path, tool_format,
                                                                                 tokenizer, monkeypatch):
    server = serve({"a": PLAYED}, token="s3cret")
    monkeypatch.setenv("DOCK_TOKEN", "s3cret")
    rows = _rows(server.url, tmp_path, tool_format, token_var="DOCK_TOKEN", attempts=1)

    played = _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer)
    monkeypatch.setenv("DOCK_TOKEN", "guess")
    refused = _generate(rows, _Engine(tokenizer, adds_a_crate(tool_format)), tokenizer)
    server.stop()

    assert [sum(reward) for reward in played["rewards"]] == [1.0]
    assert refused["env_metrics"][0]["failed"] == 1
    assert token_vars([str(tmp_path / "all.parquet"), str(tmp_path / "absent.parquet")]) == {"DOCK_TOKEN"}


def test_the_entrypoint_registers_the_env_once_and_batched_generation_and_no_env_threads_are_refused(tokenizer):
    train.register()

    with pytest.raises(ValueError, match="agentenv rows play an episode over several turns, which SkyRL's batched "
                                         "generation doesn't: set generator.batched=false"):
        _generator(_Engine(tokenizer, None), tokenizer, batched=True, chat_template_kwargs={})
    with pytest.raises(ValueError, match="set environment.skyrl_gym.max_env_workers to at least 1"):
        _generator(_Engine(tokenizer, None), tokenizer, env_workers=0)


class _Engine:
    """SkyRL's inference engine for the generator: ``script`` writes the reply to a trajectory's conversation from the
    tool answers it holds, and the reply is its tokens and the end of the turn."""

    def __init__(self, tokenizer, script):
        self.tokenizer, self.script, self.calls, self.first_prompt = tokenizer, script, 0, None

    async def generate(self, engine_input, model=None):
        self.calls += 1
        [prompt] = engine_input["prompt_token_ids"]
        self.first_prompt = self.first_prompt or list(prompt)
        reply = self.script(self.tokenizer.decode(prompt).count("<tool_response>"))
        ids = [*self.tokenizer.encode(reply, add_special_tokens=False), self.tokenizer.eos_token_id]
        return {"responses": [reply], "response_ids": [ids], "stop_reasons": ["stop"], "response_logprobs": None}

    async def finish_session(self, session_id):
        pass


def _rows(url, folder, tool_format, **options):
    return write_dataset(url, folder, tokenizer=TOKENIZERS[tool_format], tool_format=tool_format,
                         chat_template_kwargs=UNTHINKING, **options).folder / "all.parquet"


def _generator(engine, tokenizer, *, env_workers=4, **settings):
    generator_config = config.GeneratorConfig(**{"max_turns": 4, "max_input_length": 8192,
                                                 "chat_template_kwargs": UNTHINKING, **settings})
    return skyrl_generator.AgentEnvGenerator(generator_cfg=generator_config,
                                             skyrl_gym_cfg=config.SkyRLGymConfig(max_env_workers=env_workers),
                                             inference_engine_client=engine, tokenizer=tokenizer,
                                             policy_model_name="policy")


def _generate(rows, engine, tokenizer, *, samples=1, phase="train", step_wise=False, max_turns=4,
              reset_timeout=None, env_workers=4):
    prompts = dataset.PromptDataset([str(rows)], tokenizer, max_prompt_length=8192, num_workers=1)
    batch = prompts.collate_fn([prompts[index] for index in range(len(prompts.dataframe))])
    if reset_timeout is not None:
        batch = [{**row, "env_extras": {**row["env_extras"], "server": {**row["env_extras"]["server"],
                                                                        "reset_timeout": reset_timeout}}}
                 for row in batch]
    generator_input, _ = generator_utils.prepare_generator_input(batch, samples, None, "agentenv", phase, 1)
    generator = _generator(engine, tokenizer, env_workers=env_workers, max_turns=max_turns,
                           step_wise_trajectories=step_wise)
    return asyncio.run(generator.generate(generator_input))

