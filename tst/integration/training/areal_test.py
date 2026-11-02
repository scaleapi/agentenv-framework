"""``AgentEnvWorkflow`` on the opencode smoke task, as AReaL's agent workflow runs it in inline mode: opencode, in local
Docker, calls the model at the proxy address AReaL hands it, without /v1, under the sample's session key, a stand-in
model there having it run the task's shell command and then answer, and the task's verifier scores the run, with no
model endpoint configured. Needs a Docker daemon and network access for the agent's image."""

import asyncio
from pathlib import Path

import pytest

import agent_env
from agent_env.training.areal import AgentEnvWorkflow, task_rows
from tst.integration.training.harness_test import MODEL, state  # noqa: F401
from tst.util.fake_vllm import fake_vllm, shell_then_says

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]


def test_opencode_plays_its_smoke_task_against_the_proxy_address_and_the_run_returns_its_reward(state):  # noqa: F811
    shipped = Path(agent_env.__file__).parent / "agents" / "tasks" / "opencode-smoke.json"
    (state / "smoke" / "tasks").mkdir(parents=True)
    (state / "smoke" / "tasks" / shipped.name).write_text(shipped.read_text())

    with fake_vllm(MODEL, shell_then_says("echo $((6 * 7))", "42"), host="0.0.0.0") as engine:
        [row] = task_rows(state / "smoke")
        reward = asyncio.run(AgentEnvWorkflow(state / "smoke").run(row, base_url=engine.url, api_key="session-1",
                                                                   http_client=None))

    assert reward == 1.0
    assert any(message.get("role") == "tool" for body in engine.requests for message in body["messages"])
    assert {body["model"] for body in engine.requests} == {"default"}
