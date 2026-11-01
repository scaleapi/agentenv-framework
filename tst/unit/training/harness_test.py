"""``AgentEnvSessionFactory`` running a bundle's task with its own agent, whose model calls go through OpenEnv's capture
proxy to a stand-in vLLM that returns token ids and processed logprobs, as TRL's harness rollouts run it."""

import json
import logging

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_env.config.runtime import Config
from agent_env.task_step.task_step import TaskStep
from agent_env.training import harness
from agent_env.training._openenv import Server
from agent_env.training.harness import AgentEnvSessionFactory, prompt_rows
from tst.unit.bundle._support import layout
from tst.unit.conftest import local_stores  # noqa: F401

MODEL = "Qwen/Qwen3-4B"
ASKS = {"id": "asks", "type": "harness_test_asks", "verifier_id": "answer"}
asked: list[dict] = []


class _Asks(TaskStep):
    """The task's own agent, as far as the factory can tell: it asks the model its run's endpoint names, with its key and
    model, and scores the answer."""

    type = "harness_test_asks"
    entity_refs = ()

    def __init__(self, id, version, verifier_id="answer", fail=False, depends_on=None, fail_task_on_error=True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.verifier_id, self.fail = verifier_id, fail

    def to_dict(self):
        return {**super().to_dict(), "verifier_id": self.verifier_id, "fail": self.fail}

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), verifier_id=data["verifier_id"], fail=data.get("fail", False))

    async def execute(self, context):
        if self.fail:
            raise RuntimeError("the agent crashed")
        overrides = context.metadata["user_overrides"]
        async with httpx.AsyncClient() as client:
            answer = (await client.post(
                f"{overrides['litellm_base_url']}/v1/chat/completions",
                headers={"Authorization": f"Bearer {overrides['litellm_api_key']}"},
                json={"model": context.agent_model, "messages": [{"role": "user", "content": "What is 6 * 7?"}]},
            )).json()["choices"][0]["message"]["content"]
        asked.append({"model": context.agent_model, "key": overrides["litellm_api_key"], "answer": answer})
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {"score": float("42" in answer)}
        return context


@pytest.fixture
def engine():
    """A stand-in for vLLM run with --return-tokens-as-token-ids and processed logprobs."""
    requests, served = [], [MODEL]

    async def models(request):
        return JSONResponse({"data": [{"id": model, "object": "model"} for model in served]})

    async def completions(request):
        body = await request.json()
        requests.append(body)
        temperature = body.get("temperature") or 1.0
        prompt = [ord(char) % 500 + 1 for message in body["messages"] for char in str(message.get("content") or "")]
        reply = "It is 42."
        ids = [ord(char) % 500 + 600 for char in reply]
        content = [{"token": f"token_id:{token}", "logprob": -0.5 / temperature, "bytes": None,
                    "top_logprobs": [{"token": f"token_id:{token}", "logprob": -0.5 / temperature},
                                     {"token": f"token_id:{token + 1}", "logprob": -3.0 / temperature}]}
                   for token in ids]
        return JSONResponse({
            "id": "chatcmpl-1", "object": "chat.completion", "model": body["model"], "prompt_token_ids": prompt,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop",
                         "token_ids": ids, "logprobs": {"content": content}}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids), "total_tokens": len(prompt) + len(ids)},
        })

    server = Server(Starlette(routes=[Route("/v1/models", models), Route("/v1/chat/completions", completions,
                                                                          methods=["POST"])]))
    server.requests, server.served = requests, served
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Asks.type: _Asks}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)


@pytest.fixture
def bundle(tmp_path, monkeypatch, local_stores):  # noqa: F811
    monkeypatch.setenv("HOME", str(tmp_path))
    asked.clear()
    return layout(tmp_path / "quiz", {"tasks/sum.json": json.dumps([ASKS]),
                                      "tasks/broken.json": json.dumps([{**ASKS, "fail": True}])})


def test_the_rows_name_their_tasks_and_a_session_runs_its_task_with_its_calls_captured_and_scored(engine, bundle):
    rows = prompt_rows(bundle)
    factory = AgentEnvSessionFactory(bundle, llm_url=engine.url, model=MODEL, sampling={"temperature": 1.0})
    engine.served.append("trl-policy-v3")
    factory.model = "trl-policy-v3"
    session = factory.create(next(row["prompt"] for row in rows if row["task_id"].endswith("sum")))
    factory.model = "trl-policy-v4"

    assert session.wait_for_completion() == 0
    trace = session.fetch_training_trace()
    verified = session.verify([])
    session.close()

    [call] = asked
    [turn] = trace.turns
    assert (call["model"], call["answer"]) == ("trl-policy-v3", "It is 42.")
    assert (verified.env_reward, verified.metrics) == (1.0, {"score/asks": 1.0})
    assert turn.completion_token_ids == [ord(char) % 500 + 600 for char in "It is 42."]
    assert turn.loss_mask == [0] * len(turn.prompt_token_ids) + [1] * len(turn.completion_token_ids)
    assert engine.requests[-1]["model"] == "trl-policy-v3"
    assert factory._proxy.registry.get(call["key"]) is None


def test_a_run_that_fails_is_unscored_rather_than_zero(engine, bundle, caplog):
    rows = prompt_rows(bundle)
    factory = AgentEnvSessionFactory(bundle, llm_url=engine.url, model=MODEL, sampling={"temperature": 1.0})
    session = factory.create(next(row["prompt"] for row in rows if row["task_id"].endswith("broken")))

    with caplog.at_level(logging.WARNING, logger=harness.__name__):
        assert session.wait_for_completion() == 1
    assert session.verify([]).env_reward is None
    assert "failed, so its rollout is left unscored: " in caplog.text


def test_a_prompt_naming_no_task_of_the_factorys_is_refused(engine, bundle):
    factory = AgentEnvSessionFactory(bundle, llm_url=engine.url, model=MODEL)

    for prompt in ([{"role": "user", "content": "What is 6 * 7?"}],
                   [{"role": "system", "content": "AgentEnv task 0: another-bundle/sum"}]):
        with pytest.raises(KeyError, match="build the dataset with prompt_rows"):
            factory.create(prompt)


def test_an_engine_without_token_ids_is_refused_before_any_rollout(bundle, monkeypatch):
    def eval_only(llm_url, model, **_):
        raise RuntimeError(f"{llm_url} is eval-only: no token ids")

    monkeypatch.setattr(harness, "require_llm", eval_only)

    with pytest.raises(RuntimeError, match="eval-only"):
        AgentEnvSessionFactory(bundle, llm_url="http://127.0.0.1:9", model=MODEL)
