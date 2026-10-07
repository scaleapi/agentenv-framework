"""prompt_agent sends the agent an HTTPS URL for each file part naming an object the store owns, on every turn,
while the run records the object's own URL."""

import httpx
import pytest
from agentenv_protocol.a2a_agent import STAGING_V1_URI

from agent_env.config import configure
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.fake_a2a import FakeA2AAgents
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore

AGENT_URL = "http://agent.test"
USER_URL = "http://user.test"
DONE = [{"kind": "text", "text": "done"}]
USER_DONE = [{"kind": "text", "text": '{"message": "thanks", "done": true}'}]


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


def _replying_with(url):
    return [{"kind": "text", "text": "here it is"}, _file(url)]


def _two_turns(*files):
    return PromptAgentTaskStep(
        id="solve", version=None, agent_name="solver", poll_interval_seconds=0, max_conversation_turns=2,
        parts=[{"kind": "text", "text": "hi"}, *(_file(url) for url in files)],
    )


def _user_sim(url, sandbox_type="local", card=None):
    return DeployedAgent(
        agent_name="human_agent", api_url=url, a2a_url=url, sandbox_id="sb-user", sandbox_type=sandbox_type, a2a_card=card,
    )


@pytest.mark.asyncio
async def test_a_user_sim_agent_is_sent_a_file_the_target_passes_on_as_a_grant(monkeypatch, store, recorded):
    url = store.put("inputs/z.png", b"png")
    agents = FakeA2AAgents({AGENT_URL: _replying_with(url), USER_URL: USER_DONE}).serve(monkeypatch)
    context = _context()
    context.deployed_agents.append(_user_sim(USER_URL))

    await _two_turns(url).execute(context)

    assert agents.sent[USER_URL] == [
        [{"kind": "text", "text": "here it is"}, _file(f"{GRANT_ORIGIN}/inputs/z.png?sig=read")]
    ]


@pytest.mark.asyncio
async def test_a_user_sim_is_not_made_able_to_read_a_file_the_target_was_never_sent(monkeypatch, store, recorded):
    sent = store.put("inputs/z.png", b"png")
    other = store.put("someone-elses/answer.json", b"{}")
    agents = FakeA2AAgents({AGENT_URL: _replying_with(other), USER_URL: USER_DONE}).serve(monkeypatch)
    context = _context()
    context.deployed_agents.append(_user_sim(USER_URL))

    await _two_turns(sent).execute(context)

    assert agents.sent[USER_URL] == [[{"kind": "text", "text": "here it is"}, _file(other)]]
    assert store.granted == [sent]  # the target's own file, and nothing for the user-sim


@pytest.mark.asyncio
async def test_a_user_sim_the_stores_grants_cannot_reach_is_sent_a_copy_staged_on_it(monkeypatch, store, recorded):
    url = store.put("inputs/z.png", b"png")
    user_url = "https://user.test"
    staging = []  # each staging request, with how many messages the user-sim had been sent by then

    def staging_route(request):
        staging.append((request.method, len(agents.sent[user_url])))
        return httpx.Response(201 if request.method == "PUT" else 204)

    agents = FakeA2AAgents({AGENT_URL: _replying_with(url), user_url: USER_DONE}, other=staging_route).serve(monkeypatch)
    context = _context()
    card = {"capabilities": {"extensions": [{"uri": STAGING_V1_URI, "params": {"endpoint": "/ext/staging"}}]}}
    context.deployed_agents.append(_user_sim(user_url, "modal", card))

    await _two_turns(url).execute(context)

    ((text, file),) = agents.sent[user_url]
    assert file["file"]["uri"].startswith(f"{user_url}/ext/staging/")
    assert staging == [("PUT", 0), ("DELETE", 1)]  # staged before the message went, cleared once it was answered


@pytest.mark.asyncio
async def test_a_user_sim_that_can_be_sent_no_readable_url_fails_the_step_before_it_is_sent_anything(
    monkeypatch, store, recorded
):
    url = store.put("inputs/z.png", b"png")
    agents = FakeA2AAgents({AGENT_URL: _replying_with(url), USER_URL: USER_DONE}).serve(monkeypatch)
    context = _context()
    context.deployed_agents.append(_user_sim(USER_URL, "modal"))

    with pytest.raises(RuntimeError, match="cannot be sent to the agent"):
        await _two_turns(url).execute(context)

    assert agents.sent[USER_URL] == []
    assert recorded == [[{"kind": "text", "text": "hi"}, _file(url)]]  # the solver's turn, which ran


@pytest.mark.asyncio
async def test_a_registered_human_peer_is_sent_the_objects_own_url(monkeypatch, store, recorded):
    url = store.put("outputs/z.png", b"png")
    agents = FakeA2AAgents({AGENT_URL: _replying_with(url), USER_URL: DONE}).serve(monkeypatch)
    context = _context()
    context.deployed_agents.append(DeployedAgent(agent_name="human_agent", api_url=USER_URL, a2a_url=USER_URL))
    step = PromptAgentTaskStep(
        id="solve", version=None, agent_name="solver", prompt="hi", poll_interval_seconds=0, max_conversation_turns=2,
    )

    await step.execute(context)

    assert agents.sent[USER_URL] == [[{"kind": "text", "text": "here it is"}, _file(url)]]
    assert store.granted == []


@pytest.mark.asyncio
async def test_a_humans_hub_is_sent_the_objects_own_url(monkeypatch, store, recorded):
    url = store.put("outputs/z.png", b"png")
    agents = FakeA2AAgents({AGENT_URL: _replying_with(url), USER_URL: DONE}).serve(monkeypatch)
    step = PromptAgentTaskStep(
        id="solve", version=None, agent_name="solver", prompt="hi", poll_interval_seconds=0,
        max_conversation_turns=2, user_a2a_url=USER_URL,
    )

    await step.execute(_context())

    assert agents.sent[USER_URL] == [[{"kind": "text", "text": "here it is"}, _file(url)]]
    assert store.granted == []
