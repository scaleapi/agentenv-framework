"""prompt_agent sends the agent an HTTPS URL for each file part naming an object the store owns, on every turn,
while the run records the object's own URL."""

import pytest

from agent_env.config import configure
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.fake_a2a import FakeA2AAgents
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore

AGENT_URL = "http://agent.test"
USER_URL = "http://user.test"
DONE = [{"kind": "text", "text": "done"}]


@pytest.fixture
def store(tmp_path):
    granting = GrantingObjectStore(str(tmp_path))
    configure(object_store=granting)
    return granting


@pytest.fixture
def recorded(monkeypatch):
    """The parts the run records of each turn."""
    turns = []
    monkeypatch.setattr(pa.conversation_store, "add_a2a_task", lambda *a, parts, **kw: turns.append(parts))
    for fn in ("create_conversation", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa.conversation_store, fn, lambda *a, **kw: None)
    return turns


def _file(uri):
    return {"kind": "file", "file": {"uri": uri, "mimeType": "image/png", "name": "x.png"}}


def _context(sandbox_type="local"):
    context = TaskStepContext(instance_id="ti-1")
    context.metadata["seed"] = {"seed_id": "s1"}
    context.deployed_agents.append(DeployedAgent(
        agent_name="solver", api_url=AGENT_URL, a2a_url=AGENT_URL, sandbox_type=sandbox_type,
    ))
    return context


@pytest.mark.asyncio
async def test_the_agent_is_sent_a_grant_and_the_run_records_the_objects_own_url(monkeypatch, store, recorded):
    url = store.put("seeds/s1/x.png", b"png")
    agents = FakeA2AAgents({AGENT_URL: DONE}).serve(monkeypatch)
    step = PromptAgentTaskStep(
        id="solve", version=None, agent_name="solver", poll_interval_seconds=0,
        parts=[{"kind": "text", "text": "look"}, _file(store.object_url("seeds/<seed_id>/x.png"))],
    )

    result = await step.execute(_context())

    assert agents.sent[AGENT_URL] == [
        [{"kind": "text", "text": "look"}, _file(f"{GRANT_ORIGIN}/seeds/s1/x.png?sig=read")]
    ]
    assert recorded[0] == [{"kind": "text", "text": "look"}, _file(url)]
    assert result.prompt_responses[-1].source_agent_per_turn_prompt_parts == [
        [{"kind": "text", "text": "look"}, _file(url)]
    ]


@pytest.mark.asyncio
async def test_a_file_part_the_user_forwards_is_sent_as_a_grant_too(monkeypatch, store, recorded):
    url = store.put("uploads/y.png", b"png")
    forwarded = [{"kind": "text", "text": "and this"}, _file(url)]
    agents = FakeA2AAgents({AGENT_URL: DONE, USER_URL: forwarded}).serve(monkeypatch)
    step = PromptAgentTaskStep(
        id="solve", version=None, agent_name="solver", prompt="hi", poll_interval_seconds=0,
        max_conversation_turns=2, user_a2a_url=USER_URL,
    )

    result = await step.execute(_context())

    assert agents.sent[AGENT_URL][1] == [
        {"kind": "text", "text": "and this"}, _file(f"{GRANT_ORIGIN}/uploads/y.png?sig=read")
    ]
    assert result.prompt_responses[-1].source_agent_per_turn_prompt_parts[1] == forwarded


@pytest.mark.asyncio
async def test_an_object_the_agent_could_not_read_is_never_sent(monkeypatch, store, recorded):
    url = store.put("seeds/s1/x.png", b"png")
    agents = FakeA2AAgents({AGENT_URL: DONE}).serve(monkeypatch)
    step = PromptAgentTaskStep(
        id="solve", version=None, agent_name="solver", poll_interval_seconds=0, parts=[_file(url)],
    )

    with pytest.raises(RuntimeError, match="cannot be sent to the agent"):
        await step.execute(_context(sandbox_type="modal"))

    assert agents.sent[AGENT_URL] == []
    assert recorded == []  # no turn left waiting for a reply
