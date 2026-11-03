"""``AgentEnvSessionFactory`` on the shipped agents' smoke tasks, as TRL's harness rollouts run them: the harness, in
local Docker, calls the model through OpenEnv's capture proxy, in front of a stand-in vLLM that has it run the task's
shell command and then answer, so the rollout is captured as trainable turns and scored by the task's verifier, with no
model endpoint configured. Needs a Docker daemon and network access for the agents' images."""

from pathlib import Path

import pytest

import agent_env
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.training.harness import AgentEnvSessionFactory, prompt_rows
from tst.util.fake_vllm import fake_vllm, shell_then_says

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

MODEL = "Qwen/Qwen3-4B"


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder and no model endpoint; no rollout may leave a sandbox work folder."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    for name in ("LITELLM_API_KEY", "LITELLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    configure()
    reset_artifact_store()
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a rollout left a sandbox work folder"
    finally:
        reset_artifact_store()
        reset_config()


@pytest.mark.parametrize("agent", ["opencode", "claude-code"])
def test_a_harness_runs_its_smoke_task_through_the_capture_proxy_into_trainable_turns(state, agent):
    shipped = Path(agent_env.__file__).parent / "agents" / "tasks" / f"{agent}-smoke.json"
    (state / "smoke" / "tasks").mkdir(parents=True)
    (state / "smoke" / "tasks" / shipped.name).write_text(shipped.read_text())

    with fake_vllm(MODEL, shell_then_says("echo $((6 * 7))", "42")) as engine:
        [row] = prompt_rows(state / "smoke")
        factory = AgentEnvSessionFactory(state / "smoke", llm_url=engine.url, model=MODEL,
                                         sampling={"temperature": 1.0})
        session = factory.create(row["prompt"])
        try:
            assert session.wait_for_completion(900) == 0
            trace = session.fetch_training_trace()
            verified = session.verify([])
        finally:
            session.close()

    assert verified.env_reward == 1.0
    assert any(message.get("role") == "tool" for body in engine.requests for message in body["messages"])
    assert len(trace.turns) >= 2
    for turn in trace.turns:
        completion = len(turn.completion_token_ids)
        assert completion and len(turn.per_token_logps) == completion
        assert turn.loss_mask == [0] * len(turn.prompt_token_ids) + [1] * completion
