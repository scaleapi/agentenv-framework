"""A played agent's turn reaches the host with the usage of the model calls its trainer recorded: an episode's proxy,
served in this process, sums an OpenAI-shaped and an Anthropic-shaped call into one ``Usage`` whose ``input_tokens``
counts the cache, and the host stores it with the agent's reply in the run's conversation. Local stores, no Docker."""

import pytest
from agentenv_protocol.a2a_agent.tasks.v1 import Usage

from agent_env.a2a_agent import conversation_store
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.task import EpisodeSpec, Task

pytestmark = [pytest.mark.integration]

OPENAI = {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127, "prompt_tokens_details": {"cached_tokens": 100}}
ANTHROPIC = {"input_tokens": 3, "output_tokens": 9, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    configure()
    reset_artifact_store()
    try:
        yield
    finally:
        reset_config()
        reset_artifact_store()


def _stored_usage(context_id: str) -> list[dict]:
    """The usage of each agent message the run's conversation holds."""
    conversation = conversation_store.get_conversation(context_id)
    return [part["data"]["usage"] for message in conversation["messages"] if message["role"] == "agent"
            for part in message["parts"] if part.get("kind") == "data" and "usage" in (part.get("data") or {})]


@pytest.mark.asyncio
async def test_a_turns_recorded_model_calls_reach_the_host_as_the_turns_usage(local_backends):
    task = Task.from_dict({"id": "usage", "steps": [
        {"id": "policy", "type": "deploy_agent", "agent_name": "policy", "env_ids": []},
        {"id": "play", "type": "prompt_agent", "agent_name": "policy", "prompt_id": "play", "prompt": "Say hi.",
         "timeout_seconds": 60},
    ]})
    episode = await EpisodeSpec.from_task(task, trainer_agents=["policy"]).start()
    turn = await episode.next_turn()
    await turn.record_model_call({"role": "assistant", "content": "Thinking."}, model="policy-model", usage=OPENAI)
    await turn.record_model_call("Hi.", model="policy-model", usage=ANTHROPIC)
    await turn.end("Hi.")
    await episode.result()

    record = Task.get_instance(episode.id)
    assert record.status == "completed"
    [play] = record.context["prompt_responses"]
    assert play["response"] == "Hi."
    [usage] = _stored_usage(play["a2a_context_id"])
    assert Usage.from_dict(usage) == Usage(tool_call_count=0, input_tokens=120 + 3 + 100 + 20, output_tokens=16,
                                           total_tokens=127 + 123 + 9, cache_read_tokens=200, cache_write_tokens=20,
                                           input_includes_cache=True)
