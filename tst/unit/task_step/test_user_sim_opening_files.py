"""The user's first message with files (FD-5494), on the parents' visit task with an itinerary PDF: deploy_user_sim_agent
takes ``opening_prompt`` as text and file parts and tells the user sim what it attached; always_on sends the files on
the first wake in their place and the graded conversation has them; prompt_agent's match check compares the files;
the seed fills a file's address; a text-only opening is unchanged."""

from __future__ import annotations

import copy
import json

import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.a2a_agent.object_transfer import readable_parts
from agent_env.config import configure
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY, DeployUserSimAgentTaskStep
from agent_env.task_step.task_steps.agent.user_sim_common import attached, file_name, for_agent
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.always_on_harness import Rig
from tst.unit.task_step.test_deploy_user_sim_agent import HANNAH, PERSON, PROMPT, _Deploy, _published, _step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import USER, _context, _full, _reads, said
from tst.unit.task_step.test_prompt_agent_user_side import AGENT, Peers, exchange, text
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore

TEXT = PERSON["opening"]["text"]
SEED = {"trip": "nov-19"}
ITINERARY = {"kind": "file", "file": {"uri": "s3://reyes-family/<trip>/itinerary.pdf", "name": "itinerary.pdf",
                                      "mimeType": "application/pdf"},
             "metadata": {"description": "Mom and Dad's flight itinerary"}}
SHORTLIST = {"kind": "file", "file": {"uri": "s3://reyes-family/<trip>/hotel-shortlist.xlsx",
                                      "name": "hotel-shortlist.xlsx"}}
OPENING = [{"kind": "text", "text": TEXT}, ITINERARY, SHORTLIST]
NOTE = "(you attached itinerary.pdf: Mom and Dad's flight itinerary) (you attached hotel-shortlist.xlsx)"


def _sent(part: dict) -> dict:
    """A file part as the agent is sent it this run: the seed filled in, the description dropped."""
    file = {**part["file"], "uri": part["file"]["uri"].replace("<trip>", SEED["trip"])}
    return {"kind": "file", "file": file}


def _seeded() -> TaskStepContext:
    context = TaskStepContext()
    context.metadata["seed"] = dict(SEED)
    return context


# ---- deploy_user_sim_agent ------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_text_only_opening_is_recorded_and_prompted_as_before(monkeypatch):
    record = await _published(monkeypatch, _step())
    assert record["opening"] == TEXT and "opening_parts" not in record
    deploy = _Deploy(monkeypatch)
    await deploy.run(_step(opening_prompt=[{"kind": "text", "text": TEXT}]))
    assert deploy.fields("system_prompt") == [PROMPT]


@pytest.mark.asyncio
async def test_the_prompt_says_what_the_user_attached_by_description_else_by_name(monkeypatch):
    deploy = _Deploy(monkeypatch)
    context = await deploy.run(_step(opening_prompt=OPENING), _seeded())

    assert PROMPT.count(f'"{TEXT}"\n') == 1
    assert deploy.fields("system_prompt") == [PROMPT.replace(f'"{TEXT}"\n', f'"{TEXT}" {NOTE}\n')]
    record = context.metadata[USER_SIMS_KEY]["hannah"]
    # The text as before, so text readers keep working; the parts as written, the seed still to fill in.
    assert record["opening"] == TEXT
    assert record["opening_parts"] == OPENING


@pytest.mark.asyncio
async def test_the_seed_fills_a_file_s_name_in_what_the_user_sim_is_told(monkeypatch):
    deploy = _Deploy(monkeypatch)
    named = {**ITINERARY, "file": {**ITINERARY["file"], "name": "<trip>-itinerary.pdf"}}
    await deploy.run(_step(opening_prompt=[OPENING[0], named]), _seeded())
    [prompt] = deploy.fields("system_prompt")
    assert f'"{TEXT}" (you attached nov-19-itinerary.pdf: Mom and Dad\'s flight itinerary)\n' in prompt


def test_a_step_with_an_opening_of_parts_round_trips():
    step = _step(opening_prompt=OPENING)
    assert DeployUserSimAgentTaskStep.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert step.to_dict()["opening_prompt"] == OPENING


@pytest.mark.parametrize("opening, problem", [
    ([ITINERARY], "opening_prompt needs a text part with text"),
    ([{"kind": "text", "text": "  "}, ITINERARY], "opening_prompt needs a text part with text"),
    ([], "opening_prompt needs a text part with text"),
    ({"text": TEXT}, "opening_prompt must be text, or a list of text and file parts"),
    ([{"kind": "text", "text": TEXT}, "itinerary.pdf"], "opening_prompt[1] must be a part, an object"),
    ([{"kind": "text", "text": TEXT}, {"kind": "data", "data": {"flight": "UA 1534"}}],
     "opening_prompt[1] is a 'data' part; the user's first message takes text and file parts"),
    ([{"kind": "text", "text": TEXT}, {"kind": "file", "file": {"name": "itinerary.pdf"}}],
     "opening_prompt[1] is not a valid A2A file part (file.FileWithBytes.bytes: Field required; "
     "file.FileWithUri.uri: Field required)"),
    ([{"kind": "text"}], "opening_prompt[0] is not a valid A2A text part (text: Field required)"),
    ([{"kind": "text", "text": TEXT}, {**ITINERARY, "metadata": {"description": "Mom and Dad's\nflights"}}],
     "opening_prompt[1]'s metadata.description must be one line of text"),
    ([{"kind": "text", "text": TEXT}, {**ITINERARY, "metadata": {"description": " "}}],
     "opening_prompt[1]'s metadata.description must be one line of text"),
    ([{"kind": "text", "text": TEXT}, {**ITINERARY, "metadata": {"description": ["flights"]}}],
     "opening_prompt[1]'s metadata.description must be one line of text"),
], ids=["no-text", "blank-text", "empty", "not-a-list", "not-a-part", "data-part", "file-without-address",
        "text-without-text", "two-line-description", "blank-description", "description-not-text"])
def test_a_bad_opening_is_refused_when_the_task_loads(opening, problem):
    with pytest.raises(ValueError, match="deploy_user_sim_agent 'hannah'") as e:
        _step(opening_prompt=opening)
    assert problem in str(e.value)


def test_a_file_is_named_by_its_name_else_its_address_and_only_its_description_is_kept_from_the_agent():
    assert file_name({"kind": "file", "file": {"uri": "s3://reyes-family/scans/boarding-pass.png"}}) \
        == "boarding-pass.png"
    assert file_name({"kind": "file", "file": {"bytes": "aGk="}}) == "a file"
    assert attached(OPENING) == [{"name": "itinerary.pdf", "description": "Mom and Dad's flight itinerary"},
                                 {"name": "hotel-shortlist.xlsx"}]
    tagged = {**ITINERARY, "metadata": {**ITINERARY["metadata"], "source": "intake"}}
    assert for_agent(tagged) == {**ITINERARY, "metadata": {"source": "intake"}}
    assert for_agent(ITINERARY) == {"kind": "file", "file": ITINERARY["file"]}
    assert for_agent(SHORTLIST) is SHORTLIST


def test_a_read_shows_what_the_user_attached_after_their_line():
    person = {**PERSON, "opening": None}
    sent = {"at": "2026-11-16T15:50:00Z", "text": TEXT,
            "files": [{"name": "itinerary.pdf", "description": "Mom and Dad's flight itinerary"},
                      {"name": "hotel-shortlist.xlsx"}]}
    read = {"time": "2026-11-16T16:05:00Z", "assistant_messages": [{"at": "2026-11-16T16:05:00Z", "text": "On it."}],
            "phone": [], "you_sent": [sent], "done_ends_run": False}
    without = {**read, "you_sent": [{k: v for k, v in sent.items() if k != "files"}]}
    assert user_sim_versions.render(person, [], read) == user_sim_versions.render(person, [], without).replace(
        f"  [7:50 AM] {TEXT}\n", f"  [7:50 AM] {TEXT} {NOTE}\n")


# ---- always_on ----------------------------------------------------------------------------------------------------
async def _always_on_context(monkeypatch, rig: Rig, opening) -> TaskStepContext:
    """The Rig's world with Hannah as deploy_user_sim_agent records her, her first message ``opening``."""
    record = await _published(monkeypatch, _step(agent_name="user-sim", opening_prompt=opening, multi_day=False,
                                                 persona={**HANNAH, "time_zone": "America/Los_Angeles"}), _seeded())
    ctx = rig.context()
    ctx.metadata[USER_SIMS_KEY] = {"user-sim": record}
    ctx.metadata["seed"] = dict(SEED)
    ctx.metadata["clock_configurations"] = [{"step_id": "clock", "timezone": "America/Los_Angeles"}]
    return ctx


def _always_on() -> ao.AlwaysOnTaskStep:
    return ao.AlwaysOnTaskStep("week", None, user_sim="user-sim", prompt_id="ask", agent_name="solver")


def _asks_once(n, texts, rig):
    if n == 1:
        rig.message("Booking their flights now. Any seat preference?")
        return "sent"
    rig.quiet()
    return ""


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_always_on_sends_the_files_on_the_first_wake_in_their_place_and_the_chat_and_the_user_sim_have_them(
        monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, plugin=user_sim_versions, t0="2026-11-16T15:50:00Z",
              user_texts=['{"outcome": "done", "message": "aisle for dad, thanks!", "used": "none"}'])
    await rig.arm()
    later = {"kind": "text", "text": "the hotels i looked at are in the sheet"}
    ctx = await rig.run(_always_on(), await _always_on_context(monkeypatch, rig, [*OPENING[:2], later, SHORTLIST]))

    line = f"[Mon Nov 16, 3:50 PM UTC] Hannah: {TEXT}\n{later['text']}"
    opening = rig.parts[0]
    assert [p.get("text") for p in opening if p.get("kind") == "text"].count(line) == 1
    at = next(i for i, p in enumerate(opening) if p.get("text") == line)
    assert opening[at + 1:] == [_sent(ITINERARY), _sent(SHORTLIST)]  # in place, the seed filled in, no description
    [first_turn, *_] = [m for m in rig.chat(ctx) if m["role"] == "user"]
    assert [p.get("kind") for p in first_turn["parts"]] == ["text", "file", "file"]
    assert first_turn["parts"][1:] == [_sent(ITINERARY), _sent(SHORTLIST)]
    [call] = rig.user_calls
    assert (f"\nYOU SENT YOUR AI ASSISTANT:\n  [7:50 AM] {TEXT}\n{later['text']} {NOTE}\n"
            in call["parts"][0]["text"])


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_always_on_with_a_text_only_opening_sends_and_shows_only_the_text(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, plugin=user_sim_versions, t0="2026-11-16T15:50:00Z",
              user_texts=['{"outcome": "done", "message": "aisle for dad, thanks!", "used": "none"}'])
    await rig.arm()
    ctx = await rig.run(_always_on(), await _always_on_context(monkeypatch, rig, TEXT))

    assert {p.get("kind") for p in rig.parts[0]} == {"text"}
    [first_turn, *_] = [m for m in rig.chat(ctx) if m["role"] == "user"]
    assert first_turn["texts"] == [f"[Mon Nov 16, 3:50 PM UTC] Hannah: {TEXT}"]
    [call] = rig.user_calls
    assert f"\nYOU SENT YOUR AI ASSISTANT:\n  [7:50 AM] {TEXT}\n\nNEW FROM" in call["parts"][0]["text"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_an_opening_file_the_store_owns_reaches_the_agent_as_a_grant_like_any_prompt_file(monkeypatch, tmp_path):
    store = GrantingObjectStore(str(tmp_path / "objects"), reaches=True)
    configure(object_store=store)
    url = store.put("trips/nov-19/itinerary.pdf", b"%PDF")
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, plugin=user_sim_versions, t0="2026-11-16T15:50:00Z",
              user_texts=['{"outcome": "done", "message": "aisle for dad, thanks!", "used": "none"}'])
    scripted = ao.send_and_wait

    async def send_and_wait(a2a_url, parts, *, agent=None, timeout_seconds=600, **kw):
        # The real send_and_wait's sharing, in front of the Rig's scripted agent.
        if agent is None:
            return await scripted(a2a_url, parts, timeout_seconds=timeout_seconds, **kw)
        async with readable_parts(parts, a2a_url=a2a_url, card=agent.a2a_card, sandbox_type=agent.sandbox_type,
                                  expires_in=timeout_seconds) as sent:
            return await scripted(a2a_url, sent, agent=agent, timeout_seconds=timeout_seconds, **kw)

    monkeypatch.setattr(ao, "send_and_wait", send_and_wait)
    await rig.arm()
    owned = {**ITINERARY, "file": {**ITINERARY["file"], "uri": store.object_url("trips/<trip>/itinerary.pdf")}}
    ctx = await rig.run(_always_on(), await _always_on_context(monkeypatch, rig, [OPENING[0], owned]))

    assert rig.parts[0][-1] == {"kind": "file", "file": {**ITINERARY["file"],
                                                         "uri": f"{GRANT_ORIGIN}/trips/nov-19/itinerary.pdf?sig=read"}}
    [first_turn, *_] = [m for m in rig.chat(ctx) if m["role"] == "user"]
    assert first_turn["parts"][-1]["file"]["uri"] == url  # the conversation keeps the object's own address


# ---- prompt_agent with the built-in user sim ----------------------------------------------------------------------
def _prompt_agent(parts: list[dict]) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="talk", version=None, agent_name="solver", user_agent_name="hannah", parts=parts,
                               poll_interval_seconds=0, max_conversation_turns=3)


async def _hannah(monkeypatch, opening) -> TaskStepContext:
    ctx = await _context(monkeypatch, opening_prompt=opening)
    ctx.metadata["seed"] = dict(SEED)
    return ctx


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_prompt_agent_sends_the_opening_s_files_and_the_user_sim_knows_what_it_attached(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("Booking their flights now.")], USER: [said("done")]})
    ctx = await _hannah(monkeypatch, OPENING)
    # The same files, without the description, which is the user sim's.
    sent = [{"kind": "text", "text": TEXT}, for_agent(ITINERARY), SHORTLIST]

    async def armed_clock(self, envs):  # one armed clock, standing still at 7:50 AM in San Francisco
        return {"env_clock": [{"env_id": "env", "env_step_id": "clock", "virtual_time": "2026-11-16T15:50:00+00:00"}]}

    monkeypatch.setattr(PromptAgentTaskStep, "_env_clock", armed_clock)
    await _prompt_agent(sent).execute(ctx)

    first = _full(_reads(ctx)[0])
    assert peers.requests(ctx)[:2] == exchange(
        AGENT, [{"kind": "text", "text": TEXT}, _sent(ITINERARY), _sent(SHORTLIST)], "<id1>", "<ctx>", "task-1")
    assert first["read"]["you_sent"][0]["files"] == [
        {"name": "itinerary.pdf", "description": "Mom and Dad's flight itinerary"}, {"name": "hotel-shortlist.xlsx"}]
    assert f"\nYOU SENT YOUR AI ASSISTANT:\n  [3:50 PM] {TEXT} {NOTE}\n" in first["input"]  # Hannah's zone: UTC


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_prompt_agent_with_the_opening_s_parts_copied_sends_and_stores_them_without_the_description(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("Booking their flights now.")], USER: [said("done")]})
    ctx = await _hannah(monkeypatch, OPENING)
    await _prompt_agent(copy.deepcopy(OPENING)).execute(ctx)  # the opening_prompt's parts, descriptions and all

    assert peers.requests(ctx)[:2] == exchange(
        AGENT, [{"kind": "text", "text": TEXT}, _sent(ITINERARY), _sent(SHORTLIST)], "<id1>", "<ctx>", "task-1")
    doc = conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["talk"])
    assert doc["messages"][0]["parts"] == [{"kind": "text", "text": TEXT}, _sent(ITINERARY), _sent(SHORTLIST)]
    assert ctx.prompt_responses[0].source_agent_per_turn_prompt_parts[0] == doc["messages"][0]["parts"]
    assert "Mom and Dad's flight itinerary" not in json.dumps([r[2] for r in peers.raw if r[1].startswith(AGENT)])
    assert _full(_reads(ctx)[0])["read"]["you_sent"][0]["files"] == [  # the user sim still knows what it attached
        {"name": "itinerary.pdf", "description": "Mom and Dad's flight itinerary"}, {"name": "hotel-shortlist.xlsx"}]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
@pytest.mark.parametrize("files, mine", [
    ([SHORTLIST, ITINERARY], "hotel-shortlist.xlsx <s3://reyes-family/nov-19/hotel-shortlist.xlsx>; itinerary.pdf "
                             "<s3://reyes-family/nov-19/itinerary.pdf> application/pdf"),
    ([ITINERARY], "itinerary.pdf <s3://reyes-family/nov-19/itinerary.pdf> application/pdf"),
    ([{**ITINERARY, "file": {**ITINERARY["file"], "uri": "s3://reyes-family/<trip>/itinerary-v2.pdf"}}, SHORTLIST],
     "itinerary.pdf <s3://reyes-family/nov-19/itinerary-v2.pdf> application/pdf; hotel-shortlist.xlsx "
     "<s3://reyes-family/nov-19/hotel-shortlist.xlsx>"),
    ([], "none"),
], ids=["order", "missing", "other-address", "text-only"])
async def test_prompt_agent_refuses_a_prompt_whose_files_aren_t_the_opening_s(monkeypatch, files, mine):
    peers = Peers(monkeypatch, {})
    ctx = await _hannah(monkeypatch, OPENING)
    with pytest.raises(ValueError) as e:
        await _prompt_agent([{"kind": "text", "text": TEXT}, *copy.deepcopy(files)]).execute(ctx)
    assert str(e.value) == (
        f"prompt_agent 'talk': its prompt's files ({mine}) aren't the files of the opening_prompt its user 'hannah' "
        "was deployed with (itinerary.pdf <s3://reyes-family/nov-19/itinerary.pdf> application/pdf; "
        "hotel-shortlist.xlsx <s3://reyes-family/nov-19/hotel-shortlist.xlsx>); give both the same files in the same "
        "order")
    assert peers.raw == []


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_prompt_agent_refuses_files_the_text_only_opening_never_had(monkeypatch):
    Peers(monkeypatch, {})
    ctx = await _hannah(monkeypatch, TEXT)
    with pytest.raises(ValueError, match=r"its prompt's files \(itinerary.pdf <s3://reyes-family/nov-19/itinerary.pdf> "
                                         r"application/pdf\) aren't the files of the opening_prompt its user 'hannah' "
                                         r"was deployed with \(none\)"):
        await _prompt_agent([{"kind": "text", "text": TEXT}, ITINERARY]).execute(ctx)
