"""``AgentEnvSessionFactory`` running a bundle's task with its own agents, as TRL's harness rollouts run it: the
solver's deploy alone is given OpenEnv's capture proxy as its model endpoint, in front of a stand-in vLLM that returns
token ids and processed logprobs, with no endpoint configured; its calls are captured and the task's verifier scores
the run."""

import json
import logging

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import preflight
from agent_env.config.runtime import Config
from agent_env.store.routing import namespace_routing
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.training import harness
from agent_env.training._openenv import Server
from agent_env.training.harness import AgentEnvSessionFactory, prompt_rows
from tst.unit.bundle._support import layout
from tst.unit.conftest import local_stores  # noqa: F401

MODEL = "Qwen/Qwen3-4B"
MODEL_ENDPOINT_MISSING = preflight.model_endpoint_missing  # the probe, which the unit tier's conftest stubs
DEPLOY = {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "theirs"}
ASK = {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": "What is 6 * 7?", "depends_on": ["agent"]}
CHECK = {"id": "check", "type": "harness_test_check", "verifier_id": "answer", "depends_on": ["ask"]}
JUDGE = {"id": "judge", "type": "rubrics_verifier", "verifier_id": "judge", "prompt_id": "ask",
         "agent_name": DeployAgentTaskStep.DEFAULT_AGENT_NAME, "criteria": [{"id": "c1", "description": "says 42"}],
         "depends_on": ["ask"]}
asked: list[dict] = []


class _Deploys(DeployAgentTaskStep):
    """Deploys nothing: records the model endpoint the run gives the agent."""

    async def execute(self, context):
        if self.a2a_agent_id == "crashes":
            raise RuntimeError("the agent crashed")
        context.metadata["endpoint"] = self._model_endpoint(context)
        return context


class _Asks(PromptAgentTaskStep):
    """The deployed agent's turn: it asks the model at its endpoint, with its key and the run's model."""

    async def execute(self, context):
        endpoint = context.metadata["endpoint"]
        async with httpx.AsyncClient() as client:
            answer = (await client.post(
                f"{endpoint['LITELLM_BASE_URL']}/v1/chat/completions",
                headers={"Authorization": f"Bearer {endpoint['LITELLM_API_KEY']}"},
                json={"model": context.agent_model, "messages": [{"role": "user", "content": self.prompt}]},
            )).json()["choices"][0]["message"]["content"]
        asked.append({"model": context.agent_model, "key": endpoint["LITELLM_API_KEY"], "answer": answer,
                      "overrides": context.metadata["user_overrides"]})
        context.metadata["answer"] = answer
        return context


class _Checks(TaskStep):
    type = "harness_test_check"
    entity_refs = ()

    def __init__(self, id, version, verifier_id="answer", depends_on=None, fail_task_on_error=True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.verifier_id = verifier_id

    def to_dict(self):
        return {**super().to_dict(), "verifier_id": self.verifier_id}

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), verifier_id=data["verifier_id"])

    async def execute(self, context):
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "score": float("42" in context.metadata["answer"])}
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
    steps = {**Config().task_step_registry(), "deploy_agent": _Deploys, "prompt_agent": _Asks, _Checks.type: _Checks}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)


@pytest.fixture
def bundle(tmp_path, monkeypatch, local_stores):  # noqa: F811
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(preflight, "model_endpoint_missing", MODEL_ENDPOINT_MISSING)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    asked.clear()
    for agent_id in ("theirs", "crashes", "mine"):
        with namespace_routing():
            image = get_artifact_store().put_document(DockerImageArtifact(
                id=f"{agent_id}-image", description="d", image_name="ghcr.io/acme/agent:1",
                tar_gz_s3_url="s3://b/x.tar.gz"))
            A2AAgent.put(id=agent_id, docker_image_artifact=image)
    return layout(tmp_path / "quiz", {"tasks/sum.json": json.dumps([DEPLOY, ASK, CHECK]),
                                      "tasks/broken.json": json.dumps([{**DEPLOY, "a2a_agent_id": "crashes"}, ASK]),
                                      "tasks/unprompted.json": json.dumps([DEPLOY]),
                                      "tasks/self-judged.json": json.dumps([DEPLOY, ASK, JUDGE])})


def test_the_solver_alone_calls_the_policy_through_the_proxy_and_the_run_is_captured_and_scored(engine, bundle, caplog):
    with caplog.at_level(logging.WARNING, logger=harness.__name__):
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
    assert sorted(row["task_id"].rsplit("/", 1)[1] for row in rows) == ["broken", "sum"]
    assert "unprompted is left out: training needs the agent a step prompts deployed by a deploy_agent step" in caplog.text
    assert "self-judged is left out: its rubric judge is its solver, whose grading would be trained on" in caplog.text
    assert (call["model"], call["answer"]) == ("trl-policy-v3", "It is 42.")
    assert call["overrides"]["step_params"] == {"agent": {"litellm_base_url": f"http://127.0.0.1:{factory._proxy.port}",
                                                          "litellm_api_key": call["key"]}}
    assert not {"litellm_base_url", "litellm_api_key"} & call["overrides"].keys()
    assert (verified.env_reward, verified.metrics) == (1.0, {"score/check": 1.0})
    assert turn.completion_token_ids == [ord(char) % 500 + 600 for char in "It is 42."]
    assert turn.loss_mask == [0] * len(turn.prompt_token_ids) + [1] * len(turn.completion_token_ids)
    assert engine.requests[-1]["model"] == "trl-policy-v3"
    assert factory._proxy.registry.get(call["key"]) is None


def test_the_agent_named_is_the_one_the_solver_deploys(engine, bundle):
    factory = AgentEnvSessionFactory(bundle, llm_url=engine.url, model=MODEL, agent="mine")
    session = factory.create(next(row["prompt"] for row in prompt_rows(bundle, agent="mine")
                                  if row["task_id"].endswith("sum")))

    assert session.wait_for_completion() == 0
    [call] = asked
    assert call["overrides"]["step_params"]["agent"] == {
        "a2a_agent_id": "mine", "a2a_agent_version": 1, "litellm_base_url": f"http://127.0.0.1:{factory._proxy.port}",
        "litellm_api_key": call["key"]}


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


def test_a_bundle_with_no_solver_to_train_is_refused(engine, bundle):
    with pytest.raises(ValueError, match="no task of the bundle has a solver to train"):
        AgentEnvSessionFactory(bundle, llm_url=engine.url, model=MODEL, tasks=["unprompted"])


def test_an_engine_without_token_ids_is_refused_before_any_rollout(bundle, monkeypatch):
    def eval_only(llm_url, model, **_):
        raise RuntimeError(f"{llm_url} is eval-only: no token ids")

    monkeypatch.setattr(harness, "require_llm", eval_only)

    with pytest.raises(RuntimeError, match="eval-only"):
        AgentEnvSessionFactory(bundle, llm_url="http://127.0.0.1:9", model=MODEL)
