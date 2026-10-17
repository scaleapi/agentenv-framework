"""user_system_prompt: the user-sim's persona, set through its agent-config extension, for a user-sim that has
no other way to get one (e.g. installed with install_agent rather than deployed with a system_prompt)."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _card(*supported: str) -> dict:
    return {"capabilities": {"extensions": [{
        "uri": A2AAgent.EXT_AGENT_CONFIG,
        "params": {"endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": list(supported)}}}},
    }]}}


def _wire(monkeypatch):
    """Mock the conversation's HTTP and store boundary. Returns (config posts, message sends)."""
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    cfg.get_default_human_a2a_url.return_value = "http://hub/a2a/human"
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)
    posts: list[tuple[str, dict]] = []
    sends: list[str] = []

    async def post_agent_config(url, fields):
        posts.append((url, fields))

    async def send(url, parts, message_id, context_id, timeout):
        sends.append(url)
        return f"task-{len(sends)}", None

    async def poll(url, task_id, timeout, interval, **_):
        text = '{"message": "thanks", "done": true}' if "user-sim" in url else "done"
        return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": text}]}}}

    monkeypatch.setattr(pa_mod.protocol, "post_agent_config", post_agent_config)
    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)
    return posts, sends


def _context(user_card: dict | None, seed: dict | None = None) -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1", metadata={"seed": seed} if seed else {})
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                             a2a_card={}))
    if user_card is not None:
        ctx.deployed_agents.append(DeployedAgent(agent_name="usersim", api_url="http://user-sim",
                                                 a2a_url="http://user-sim", a2a_card=user_card))
    return ctx


def _step(**kw) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver",
                               max_conversation_turns=2, user_agent_name="usersim", **kw)


def test_round_trips_and_is_omitted_when_unset():
    step = PromptAgentTaskStep.from_dict({"id": "s", "version": 1, "prompt": "hi", "user_system_prompt": "You are Ana."})
    assert step.user_system_prompt == "You are Ana."
    assert step.to_dict()["user_system_prompt"] == "You are Ana."
    unset = PromptAgentTaskStep.from_dict({"id": "s", "version": 1, "prompt": "hi"})
    assert unset.user_system_prompt is None
    assert "user_system_prompt" not in unset.to_dict()


@pytest.mark.asyncio
async def test_the_user_sim_gets_its_persona_with_the_seed_applied(monkeypatch):
    posts, _ = _wire(monkeypatch)
    step = _step(user_system_prompt="You are <name>, the requester.")

    await step.execute(_context(_card("output_format", "model", "system_prompt"), seed={"name": "Ana"}))

    user_posts = [fields for url, fields in posts if url.startswith("http://user-sim")]
    assert user_posts and user_posts[0]["system_prompt"] == "You are Ana, the requester."


@pytest.mark.asyncio
async def test_without_it_the_user_sim_gets_no_system_prompt(monkeypatch):
    posts, _ = _wire(monkeypatch)

    await _step().execute(_context(_card("output_format", "model", "system_prompt")))

    user_posts = [fields for url, fields in posts if url.startswith("http://user-sim")]
    assert user_posts and "system_prompt" not in user_posts[0]


@pytest.mark.asyncio
async def test_a_user_sim_that_cannot_take_a_system_prompt_fails_before_any_turn(monkeypatch):
    _, sends = _wire(monkeypatch)
    step = _step(user_system_prompt="You are Ana.")

    with pytest.raises(RuntimeError, match="does not accept system_prompt"):
        await step.execute(_context(_card("output_format", "model")))
    assert sends == []


@pytest.mark.asyncio
async def test_it_needs_a_deployed_user_sim(monkeypatch):
    _, sends = _wire(monkeypatch)
    step = _step(user_system_prompt="You are Ana.")

    with pytest.raises(RuntimeError, match="no deployed user-sim 'usersim'"):
        await step.execute(_context(user_card=None))
    assert sends == []
