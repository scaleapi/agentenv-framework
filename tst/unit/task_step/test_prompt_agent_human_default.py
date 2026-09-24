"""The human-A2A default resolves only when a conversation will need it — unconfigured single-turn runs keep working."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.config.errors import ConfigError
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _wire(monkeypatch, cfg):
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task",
               "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)

    async def send(url, parts, message_id, context_id, timeout):
        return "task-1", None

    async def poll(url, task_id, timeout, interval):
        return {
            "status": {
                "state": TaskState.completed,
                "message": {"parts": [{"kind": "text", "text": "done"}]},
            }
        }

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)


def _context():
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents.append(
        DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver", a2a_card={})
    )
    return ctx


def _cfg():
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    return cfg


@pytest.mark.asyncio
async def test_single_turn_never_resolves_the_human_default(monkeypatch):
    cfg = _cfg()
    cfg.get_default_human_a2a_url.side_effect = AssertionError(
        "human default resolved on a single-turn run"
    )
    _wire(monkeypatch, cfg)
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver")

    result = await step.execute(_context())

    assert result.prompt_responses[-1].response == "done"
    cfg.get_default_human_a2a_url.assert_not_called()


@pytest.mark.asyncio
async def test_multi_turn_without_a_user_fails_before_any_solver_spend(monkeypatch):
    cfg = _cfg()
    cfg.get_default_human_a2a_url.side_effect = ConfigError("No human-A2A URL configured")
    _wire(monkeypatch, cfg)
    sent_urls: list[str] = []

    async def send(url, parts, message_id, context_id, timeout):
        sent_urls.append(url)
        return "task-1", None

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver", max_conversation_turns=2
    )

    with pytest.raises(ConfigError, match="No human-A2A URL configured"):
        await step.execute(_context())
    assert sent_urls == []


@pytest.mark.asyncio
async def test_multi_turn_with_a_configured_default_resolves_once(monkeypatch):
    cfg = _cfg()
    cfg.get_default_human_a2a_url.return_value = "http://hub/a2a/human"
    _wire(monkeypatch, cfg)
    sent_urls: list[str] = []

    async def send(url, parts, message_id, context_id, timeout):
        sent_urls.append(url)
        return f"task-{len(sent_urls)}", None

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver", max_conversation_turns=3
    )

    await step.execute(_context())

    assert sent_urls.count("http://hub/a2a/human") == 2
    cfg.get_default_human_a2a_url.assert_called_once()


@pytest.mark.asyncio
async def test_a_deployed_user_sim_routes_turns_to_it_not_the_human_default(monkeypatch):
    cfg = _cfg()
    cfg.get_default_human_a2a_url.side_effect = AssertionError(
        "human default resolved despite a deployed user-sim"
    )
    _wire(monkeypatch, cfg)
    sent_urls: list[str] = []

    async def send(url, parts, message_id, context_id, timeout):
        sent_urls.append(url)
        return f"task-{len(sent_urls)}", None

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    ctx = _context()
    ctx.deployed_agents.append(
        DeployedAgent(agent_name="human_agent", api_url="http://user-sim", a2a_url="http://user-sim", a2a_card={})
    )
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver", max_conversation_turns=2
    )

    await step.execute(ctx)

    assert "http://user-sim" in sent_urls
    cfg.get_default_human_a2a_url.assert_not_called()
