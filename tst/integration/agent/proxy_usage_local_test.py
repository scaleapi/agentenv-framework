"""A played agent's turn reaches the host with the usage of the model calls its trainer recorded: an episode's proxy,
served in this process, sums an OpenAI-shaped and an Anthropic-shaped response's usage and a ``Usage`` the trainer
reports itself into one ``Usage`` whose ``input_tokens`` counts the cache, and the host stores it with the agent's reply
in the run's conversation. Local stores, no Docker."""

import pytest
from agentenv_protocol.a2a_agent.tasks.v1 import Usage

from agent_env.a2a_agent import conversation_store
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.task import EpisodeSpec, Task

pytestmark = [pytest.mark.integration]

OPENAI = {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127, "prompt_tokens_details": {"cached_tokens": 100}}
ANTHROPIC = {"input_tokens": 3, "output_tokens": 9, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}
REPORTED = Usage(input_tokens=4, output_tokens=2, total_tokens=6, cache_read_tokens=10, input_includes_cache=False,
                 cost_usd=0.02)


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
    try:
        turn = await episode.next_turn()
        await turn.record_model_call({"role": "assistant", "content": "Thinking."}, model="policy-model",
                                     usage=OPENAI)
        await turn.record_model_call({"role": "assistant", "content": "Still thinking."}, model="policy-model",
                                     usage=ANTHROPIC)
        await turn.record_model_call("Hi.", model="policy-model", usage=REPORTED)
        await turn.end("Hi.")
        await episode.result()
    finally:
        await episode.abandon("the test ended")

    record = Task.get_instance(episode.id)
    assert record.status == "completed"
    [play] = record.context["prompt_responses"]
    assert play["response"] == "Hi."
    [usage] = _stored_usage(play["a2a_context_id"])
    assert Usage.from_dict(usage) == Usage(
        tool_call_count=0, input_tokens=120 + 123 + 14, output_tokens=18, total_tokens=127 + 132 + 16,
        cache_read_tokens=210, cache_write_tokens=20, input_includes_cache=True, cost_usd=0.02,
        provider_details={"model_calls": 3, "unreported": {"cache_write_tokens": 2, "cost_usd": 2}})


@pytest.mark.asyncio
async def test_a_played_agents_usage_is_never_priced_from_the_table_and_keeps_its_trainers_cost(
        local_backends, monkeypatch, tmp_path):
    """The proxy takes no model credentials, so its trainer's calls don't go through the configured endpoint: the
    table doesn't price them though it lists the model, and the stored entry's cost is the figure the trainer
    reported, as the agent's own."""
    config = tmp_path / "prices.toml"
    config.write_text('[model.prices]\n"policy-model" = { input = 1.0, output = 2.0, cache_read = 0.5 }\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    reset_config()
    configure()
    task = Task.from_dict({"id": "usage", "steps": [
        {"id": "policy", "type": "deploy_agent", "agent_name": "policy", "env_ids": []},
        {"id": "play", "type": "prompt_agent", "agent_name": "policy", "prompt_id": "play", "prompt": "Say hi.",
         "model": "policy-model", "timeout_seconds": 60},
    ]})
    episode = await EpisodeSpec.from_task(task, trainer_agents=["policy"]).start()
    try:
        turn = await episode.next_turn()
        await turn.record_model_call("Hi.", model="policy-model", usage=REPORTED)
        await turn.end("Hi.")
        await episode.result()
    finally:
        await episode.abandon("the test ended")

    record = Task.get_instance(episode.id)
    [agent] = record.context["deployed_agents"]
    [play] = record.context["prompt_responses"]
    [entry] = play["usage"]
    assert agent.get("on_configured_endpoint") is not True
    assert {key: entry.get(key) for key in ("cost_usd", "cost_status", "cost_source", "reported_cost_usd")} == {
        "cost_usd": 0.02, "cost_status": "reported", "cost_source": "harness", "reported_cost_usd": 0.02}
    assert "unit_prices" not in entry and (entry["input_tokens"], entry["cache_read_tokens"]) == (14, 10)
