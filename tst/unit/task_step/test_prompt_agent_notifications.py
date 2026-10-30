"""prompt_agent delivers what came due on its envs' notifications/v1: after each turn that the conversation goes on
from, at the start of the agent's next message, ahead of the user side's reply. The env answers take from a real
gateway queue (``Notifications``) on a frozen clock; the agent, user side and env are scripted over HTTP."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import EXT_NOTIFICATIONS_URI
from agent_env.env.gateway.notifications import Notifications
from agent_env.task_step.task_steps import notification_delivery
from agent_env.task_step.task_steps.notification_delivery import delivered_line, next_message
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.test_prompt_agent_post_turn_hooks import EnvPeers, _env, _hook
from tst.unit.task_step.test_prompt_agent_user_side import AGENT, ENV, USER, _USER_SIM_CARD, _context, sim, text

pytestmark = pytest.mark.usefixtures("local_stores")

MONDAY = "2027-02-01T15:00:00Z"  # 7:00 AM in Los Angeles
ALDER = "See if Dana replied about the 2BR on Alder St"
IVY = "New listing: 2BR on Ivy St, $3,050/mo"


class GatewayQueue(EnvPeers):
    """``EnvPeers`` whose env answers notifications/v1 take from a real ``Notifications`` on a frozen clock."""

    def __init__(self, monkeypatch, replies, *, fail_take: bool = False, **kw):
        super().__init__(monkeypatch, replies, **kw)
        self.gw_clock = Clock(now_fn=lambda: 1000.0)
        self.gw_clock.set_time(MONDAY, 0)
        self.queue = Notifications(lambda: self.gw_clock)
        self.queue.configure({"roles": ["default", "planner"]})
        self.takes: list[dict] = []
        self.fail_take = fail_take

    def _answer(self, request: httpx.Request) -> httpx.Response:
        origin = f"{request.url.scheme}://{request.url.host}"
        if origin == ENV and request.url.path == "/notifications/take":
            body = json.loads(request.content)
            assert body.pop("take_id")  # fresh per take, reused only by its own retries
            self.takes.append(body)
            if self.fail_take:
                return httpx.Response(503, json={"ok": False, "error": "gateway restarting"})
            return httpx.Response(200, json=self.queue.take(body))
        return super()._answer(request)

    def at(self, when: str) -> None:
        self.gw_clock.set_time(when, 0)


def _ctx(*, role: str | None = None, user_sim: bool = True):
    ctx = _context(user_card=_USER_SIM_CARD if user_sim else None)
    ctx.deployed_agents[0].role = role
    ctx.deployed_envs.append(_env(EXT_NOTIFICATIONS_URI))
    return ctx


def _step(turns: int = 4, **kw) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="talk", version=None, prompt="find me a 2BR", agent_name="solver",
                               poll_interval_seconds=0, max_conversation_turns=turns, **kw)


def _to_agent(ctx) -> list[list[dict]]:
    """Each message the agent was sent, in order."""
    doc = conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["talk"])
    return [m["parts"] for m in doc["messages"] if m["role"] == "user"]


def _alder_and_ivy_due(peers: GatewayQueue) -> None:
    """A check the agent scheduled for 3:00 PM and a Curbfinch alert at 3:10 PM, Los Angeles time."""
    peers.queue.schedule(ALDER, at="2027-02-01T23:00:00Z")
    peers.at("2027-02-01T23:10:00Z")
    peers.queue.notify("Curbfinch alert", IVY, trigger_id="ivy-listed")


# ---- with the user sim ----------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_what_came_due_opens_the_next_message_ahead_of_the_user_sims_reply(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("two listings so far"), text("Fremont too, noted")],
                                       USER: [sim("Actually, can we look at places in Fremont too?"),
                                              sim("great", done=True)]})
    _alder_and_ivy_due(peers)
    ctx = _ctx()

    await _step().execute(ctx)

    assert _to_agent(ctx)[1] == [
        {"kind": "text", "text": f"[Mon Feb 1, 11:00 PM UTC · your scheduled check] {ALDER}",
         "metadata": {"agentenv": {"id": "check-1", "to": "default", "at": "2027-02-01T23:00:00Z", "times": 1,
                                   "source": "scheduled_check", "source_id": "check-1",
                                   "channel": "your scheduled check", "env_id": "env"}}},
        {"kind": "text", "text": f"[Mon Feb 1, 11:10 PM UTC · Curbfinch alert] {IVY}",
         "metadata": {"agentenv": {"id": "notification-2", "to": "default", "at": "2027-02-01T23:10:00Z", "times": 1,
                                   "source": "trigger", "source_id": "ivy-listed", "channel": "Curbfinch alert",
                                   "env_id": "env"}}},
        {"kind": "text", "text": "Actually, can we look at places in Fremont too?"},
    ]
    assert peers.takes == [{"to": "default"}]  # the user sim's done ended the run: its turn took nothing


@pytest.mark.asyncio
async def test_nothing_due_leaves_every_message_as_it_was(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("draft one"), text("draft two")],
                                       USER: [sim("make it shorter"), sim("perfect", done=True)]})
    ctx = _ctx()

    await _step().execute(ctx)

    assert _to_agent(ctx) == [text("find me a 2BR"), text("make it shorter")]
    assert peers.takes == [{"to": "default"}]


@pytest.mark.asyncio
async def test_an_answer_that_ends_the_conversation_takes_nothing_and_leaves_it_queued(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("done looking")], USER: [sim("thanks", done=True)]})
    _alder_and_ivy_due(peers)
    ctx = _ctx()

    await _step().execute(ctx)

    assert peers.takes == [] and _to_agent(ctx) == [text("find me a 2BR")]
    assert peers.queue.state("default")["next_due_at"] == "2027-02-01T23:00:00Z"
    assert len(peers.queue.state("default")["pending_notifications"]) == 1


@pytest.mark.asyncio
async def test_the_last_turn_takes_nothing(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("one")]})
    _alder_and_ivy_due(peers)

    await _step(turns=1).execute(_ctx())

    assert peers.takes == []


@pytest.mark.asyncio
async def test_labels_are_in_utc_and_say_so(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("ok"), sim("bye", done=True)]})
    _alder_and_ivy_due(peers)
    ctx = _ctx()

    await _step().execute(ctx)

    assert [p["text"] for p in _to_agent(ctx)[1]] == [f"[Mon Feb 1, 11:00 PM UTC · your scheduled check] {ALDER}",
                                                      f"[Mon Feb 1, 11:10 PM UTC · Curbfinch alert] {IVY}", "ok"]


@pytest.mark.asyncio
async def test_only_what_is_due_for_the_agents_own_role_reaches_it(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("ok"), sim("bye", done=True)]})
    peers.queue.notify("Messages", "for the solver role")
    peers.queue.notify("Calendar", "for the planner", to="planner")
    ctx = _ctx(role="planner")

    await _step().execute(ctx)

    assert peers.takes == [{"to": "planner"}]
    assert [p["text"] for p in _to_agent(ctx)[1]] == ["[Mon Feb 1, 3:00 PM UTC · Calendar] for the planner", "ok"]
    assert [n["text"] for n in peers.queue.state("default")["pending_notifications"]] == ["for the solver role"]


@pytest.mark.asyncio
async def test_a_take_that_keeps_failing_fails_the_step_rather_than_drop_what_is_due(monkeypatch):
    monkeypatch.setattr(notification_delivery, "_TAKE_ATTEMPTS", 1)
    GatewayQueue(monkeypatch, {AGENT: [text("one")], USER: [sim("ok")]}, fail_take=True)

    with pytest.raises(RuntimeError, match="could not take what came due for role 'default' from env 'env'"):
        await _step().execute(_ctx())


# ---- with a post-turn hook ------------------------------------------------------------------------------------------

SAYS_NOTHING = """
    async def post_turn(turn, params):
        return None if turn.last else {"message": []}
"""
SAYS_SOMETHING = """
    async def post_turn(turn, params):
        return None if turn.last else {"message": "[Tue] the person writes"}
"""


@pytest.mark.asyncio
async def test_a_hook_that_says_nothing_sends_only_what_came_due_and_the_run_ends_when_nothing_has(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("watching"), text("on it")]})
    _alder_and_ivy_due(peers)
    ctx = _ctx(user_sim=False)

    await _step(**_hook_spec(_hook("quiet", SAYS_NOTHING))).execute(ctx)

    assert [[p["text"] for p in parts] for parts in _to_agent(ctx)] == [
        ["find me a 2BR"],
        [f"[Mon Feb 1, 11:00 PM UTC · your scheduled check] {ALDER}", f"[Mon Feb 1, 11:10 PM UTC · Curbfinch alert] {IVY}"],
    ]
    assert peers.takes == [{"to": "default"}, {"to": "default"}]  # the second found nothing, so the run ended
    doc = conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["talk"])
    assert doc["status"] == "closed"


@pytest.mark.asyncio
async def test_what_came_due_goes_ahead_of_a_hooks_message(monkeypatch):
    peers = GatewayQueue(monkeypatch, {AGENT: [text("one"), text("two")]})
    peers.queue.notify("Messages", "Jamie: did the rug thing get sorted?")
    ctx = _ctx(user_sim=False)
    await _step(turns=2, **_hook_spec(_hook("says", SAYS_SOMETHING))).execute(ctx)

    assert [p["text"] for p in _to_agent(ctx)[1]] == [
        "[Mon Feb 1, 3:00 PM UTC · Messages] Jamie: did the rug thing get sorted?", "[Tue] the person writes"]


def _hook_spec(hook_id: str) -> dict:
    return {"post_turn_hooks": [{"file_artifact_id": hook_id}]}


# ---- the step field and the message builder -------------------------------------------------------------------------

def test_time_zone_is_not_a_step_field():
    with pytest.raises(TypeError):
        _step(time_zone="America/Los_Angeles")


def _item(source: str, channel: str, text: str, at: str | None = MONDAY, times: int = 1) -> dict:
    return {"id": "x-1", "to": "default", "at": at, "times": times, "source": source, "source_id": "x",
            "channel": channel, "parts": [{"kind": "text", "text": text}]}


def test_every_item_is_labelled_with_its_channel_and_one_that_came_due_more_than_once_says_so():
    check = _item("scheduled_check", "your scheduled check", "check rugs", times=4)
    assert delivered_line(check) == (
        "[Mon Feb 1, 3:00 PM UTC · your scheduled check] check rugs (it came due 4 times while you were away)")
    queued_unclocked = _item("trigger", "Gmail", "hi", at=None)
    assert delivered_line(queued_unclocked) == "[now · Gmail] hi"
    unknown_source = _item("calendar", "Calendar", "Standup in 10 minutes")
    assert delivered_line(unknown_source) == "[Mon Feb 1, 3:00 PM UTC · Calendar] Standup in 10 minutes"


def test_a_delivered_part_carries_the_item_without_its_parts_and_a_part_that_is_not_text_follows_as_it_is():
    item = {**_item("trigger", "Gmail", "New email"), "env_id": "env"}
    item["parts"].append({"kind": "data", "data": {"thread": "t-9"}})
    meta = {"agentenv": {k: v for k, v in item.items() if k != "parts"}}
    assert notification_delivery.delivered_parts([item]) == [
        {"kind": "text", "text": "[Mon Feb 1, 3:00 PM UTC · Gmail] New email", "metadata": meta},
        {"kind": "data", "data": {"thread": "t-9"}, "metadata": meta},
    ]


def test_a_blank_reply_adds_nothing_to_delivered_lines_and_without_them_is_sent_as_it_was():
    delivered = [{"kind": "text", "text": "[now · Gmail] hi"}]
    assert next_message(delivered, [{"kind": "text", "text": "  "}]) == delivered
    assert next_message([], [{"kind": "text", "text": "  "}]) == [{"kind": "text", "text": "  "}]
    assert next_message(delivered, text("and me")) == [*delivered, *text("and me")]


class _FlakyEnv:
    """An env whose first take is applied but its response lost, as a dropped connection would."""

    env_id = "flaky"

    def __init__(self, notifications):
        self.notifications = notifications
        self.bodies = []

    def supports(self, extension, method):
        return extension == EXT_NOTIFICATIONS_URI

    async def invoke(self, extension, method, body):
        self.bodies.append(dict(body))
        answer = self.notifications.take(body)
        if len(self.bodies) == 1:
            raise RuntimeError("connection reset")
        return answer


@pytest.mark.asyncio
async def test_a_take_retried_after_a_lost_response_still_delivers_the_item(monkeypatch):
    monkeypatch.setattr(notification_delivery, "_TAKE_RETRY_BASE_DELAY_S", 0)
    clock = Clock()
    clock.set_time("2027-02-01T15:00:00Z", rate=0)
    n = Notifications(lambda: clock)
    n.notify("Messages", "Jamie: did the rug thing get sorted?")
    env = _FlakyEnv(n)
    items = await notification_delivery.take_due([env], "default")
    assert [i["parts"][0]["text"] for i in items] == ["Jamie: did the rug thing get sorted?"]
    assert env.bodies[0]["take_id"] == env.bodies[1]["take_id"]
