"""Unit tests for the gateway trigger engine (urn:agentenv:triggers/v1)."""
from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway import triggers as triggers_module
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError, _next_opening


class _FakeGateway:
    def __init__(self, responses: dict) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []
        self.roles: list[tuple[str, str | None]] = []  # the `role` each _internal_call carried
        self.events: list[dict] = []
        self.rules: list[tuple[str, str, bool]] = []
        self._role_rules_lock = asyncio.Lock()
        self._server_tools: dict = {}
        self._tool_server_urls: dict = {}
        self._clock = Clock()  # unarmed: events carry no virtual_time (the zero-delta path)
        self.TOOL_CALL_TIMEOUT_S = 30

    def _apply_rule(self, role, tool, value):
        self.rules.append((role, tool, value))

    def _query_changelog_id(self):
        return 42

    async def _log_event(self, event):
        self.events.append(dict(event))
        return f"event_{len(self.events)}"

    async def _ensure_tools_discovered(self):
        pass


def _result(text: str = "{}", is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


@pytest.fixture
def engine(monkeypatch):
    state = {"doc_text": "Overview only.", "slack": []}
    responses = {
        "gdocs_search_documents": lambda a: {"documents": [
            {"documentId": "decoy", "title": "Unrelated"},
            {"documentId": "doc-1", "title": "Probe Doc"},
        ]},
        "gdocs_get_document_text": lambda a: {"documentId": a["documentId"], "text": state["doc_text"]},
        "slack_send_message": lambda a: (state["slack"].append(a) or {"ok": True}),
        "slack_conversations_history": lambda a: {"messages": [{"text": m["text"]} for m in state["slack"]]},
    }
    gw = _FakeGateway(responses)

    async def fake_internal_call(self, tool_name, arguments, role=None):
        gw.calls.append((tool_name, dict(arguments)))
        gw.roles.append((tool_name, role))
        fn = gw.responses.get(tool_name)
        if fn is None:
            raise RuntimeError(f"unknown tool {tool_name}")
        out = fn(arguments)
        return _result(out if isinstance(out, str) else json.dumps(out))

    monkeypatch.setattr(TriggerEngine, "_internal_call", fake_internal_call)
    eng = TriggerEngine(gw)
    eng._test_state = state
    eng._test_gw = gw
    return eng


@pytest.fixture
def executor_calls(monkeypatch):
    """Every executor call as (executor, instruction, context_id); the stub answers with `.reply`, raising it if an
    exception."""
    calls = _ExecutorCalls()

    async def fake_executor(self, executor, instruction, context_id):
        calls.append((executor, instruction, context_id))
        if isinstance(calls.reply, BaseException):
            raise calls.reply
        return calls.reply

    monkeypatch.setattr(TriggerEngine, "_call_executor", fake_executor)
    return calls


@pytest.fixture
def waits(engine, monkeypatch):
    """Every virtual-time wait the engine makes, in virtual seconds, on an unarmed clock whose source only the waits
    advance, so none of them actually sleeps."""
    return _fake_waits(engine, monkeypatch)


@pytest.fixture
def virtual_time(engine, monkeypatch):
    """The engine's clock armed at 2026-10-14T08:00Z (a Wednesday) on a fake source; each wait advances it at once
    and is recorded in virtual seconds, and `.advance(s)` moves it by `s` virtual seconds."""
    return _fake_waits(engine, monkeypatch, armed_at="2026-10-14T08:00:00Z")


def _registration(**overrides):
    base = {
        "watch_roles": ["default"],
        "triggers": [
            {"id": "grant", "when": {"type": "action", "tool": "gdocs_create_document",
                                     "where": {"title": {"regex": "Probe Doc"}}},
             "actions": [{"type": "permission", "action": "enable", "role": "default", "tools": ["snowflake_x"]}]},
            {"id": "rate", "when": {"type": "state", "check": {"steps": [
                {"tool": "gdocs_search_documents", "args": {"query": "Probe Doc"},
                 "extract": {"path": "documents[*]", "match": {"title": {"regex": "Probe Doc"}},
                             "take": "documentId"},
                 "bind": "doc_id"},
                {"tool": "gdocs_get_document_text", "args": {"documentId": "${doc_id}"},
                 "extract": "text", "predicate": {"regex": "Budget Fit"}}]}},
             "actions": [{"type": "tool", "tool": "slack_send_message",
                          "args": {"channel": "#c", "text": "rate is $26/hr"},
                          "verify": {"tool": "slack_conversations_history", "args": {"channel": "#c"},
                                     "predicate": {"regex": r"\$26/hr"}}}]},
            {"id": "sensor", "when": {"type": "state", "check": {"steps": [
                {"tool": "gdocs_get_document_text", "args": {"documentId": "doc-1"},
                 "extract": "text", "predicate": {"regex": r"\$26/hr"}}]}},
             "actions": []},
        ],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad,fragment", [
    ({"triggers": "nope"}, "triggers must be a list"),
    ({"triggers": [{"id": "x", "when": {"type": "step"}, "actions": []}]}, "when.type"),
    ({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "actions": [{"type": "nl", "instruction": "hi"}]}]}, "require an executor"),
    ({"triggers": [{"id": "x", "when": {"type": "state", "check": {"steps": [
        {"tool": "t", "predicate": {"contains": "sub"}}]}}, "actions": []}]}, "exactly one of"),
    ({"triggers": [{"id": "x", "when": {"type": "state", "check": {"steps": [
        {"tool": "t", "args": {}}]}}, "actions": []}]}, "last check step"),
    ({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "where": "notadict"},
                    "actions": []}]}, "where must be an object"),
    ({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "where": {}, "actions": []},
                   {"id": "x", "when": {"type": "action", "tool": "t"}, "actions": []}]}, "duplicate id"),
    ({"executor": {"a2a_url": "http://e"}, "triggers": []}, "executor.role must be"),
    ({"watch_roles": ["default", "executor"], "executor": {"a2a_url": "http://e", "role": "executor"},
      "triggers": []}, "must not be in watch_roles"),
    *[({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                      "actions": [{"type": "tool", "tool": "t2", "as": acting}]}]}, fragment)
      for acting, fragment in [("", "'as' must be a non-empty"), ("   ", "'as' must be a non-empty"),
                               ("default", "'as' must be a non-empty"), (" default ", "'as' must be a non-empty"),
                               (5, "'as' must be a non-empty"), ("${args.owner}", "not templated")]],
    ({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "actions": [{"type": "permission", "action": "enable", "role": "r", "tools": ["t"], "as": "x"}]}]},
     "only valid on tool actions"),
    ({"executor": {"a2a_url": "http://e", "role": "executor"},
      "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "actions": [{"type": "nl", "instruction": "hi", "as": "x"}]}]}, "only valid on tool actions"),
    ({"executors": ["bob"], "triggers": []}, "executors must be an object"),
    ({"executors": {"": {"a2a_url": "http://b", "role": "bob"}}, "triggers": []}, "executors must be an object"),
    ({"executors": {"bob": {"a2a_url": "http://b"}}, "triggers": []}, "executors['bob'].role must be"),
    ({"executors": {"bob": {"role": "bob"}}, "triggers": []}, "executors['bob'].a2a_url must be"),
    ({"watch_roles": ["default", "bob"], "executors": {"bob": {"a2a_url": "http://b", "role": "bob"}},
      "triggers": []}, "must not be in watch_roles"),
    ({"executors": {"bob": {"a2a_url": "http://b", "role": "bob"}},
      "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "actions": [{"type": "nl", "instruction": "hi", "executor": "carol"}]}]},
     "executor 'carol' is not a registered executor name"),
    ({"executor": {"a2a_url": "http://e", "role": "executor"},
      "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "actions": [{"type": "nl", "instruction": "hi", "executor": "executor"}]}]},
     "is not a registered executor name"),
    ({"executors": {"bob": {"a2a_url": "http://b", "role": "bob"}},
      "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                    "actions": [{"type": "tool", "tool": "t2", "executor": "bob"}]}]}, "only valid on nl actions"),
    *[({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                      "actions": [{"type": "tool", "tool": "t2", "delay": delay}]}]}, fragment)
      for delay, fragment in [("soon", "delay"), ("PT0S", "delay must be a positive duration"),
                              ({"dist": "uniform"}, "delay must be an ISO-8601 duration or"),
                              ({"dist": "exp", "mean": "PT1M"}, "requires an integer 'seed'"),
                              ({"dist": "exp", "mean": "PT0S", "seed": 1}, "delay.mean must be a positive")]],
    ({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"}, "barrier": {"at": "provoking_call"},
                    "actions": [{"type": "tool", "tool": "t2", "delay": "PT1M"}]}]}, "can't run under a barrier"),
    ({"executors": {"bob": {"a2a_url": "http://b", "role": "bob", "hours": {"start": "09:00", "end": "17:00"}}},
      "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"}, "barrier": {"at": "provoking_call"},
                    "actions": [{"type": "nl", "executor": "bob", "instruction": "hi"}]}]}, "can't run under a barrier"),
    *[({"executors": {"bob": {"a2a_url": "http://b", "role": "bob", "hours": hours}}, "triggers": []}, fragment)
      for hours, fragment in [({"start": "9:00", "end": "17:00"}, "hours.start must be a 24-hour"),
                              ({"start": "09:00", "end": "09:00"}, "would never be available"),
                              ({"start": "09:00", "end": "17:00", "utc_offset": "PST"}, "utc_offset must look like"),
                              ({"start": "09:00", "end": "17:00", "days": ["monday"]}, "hours.days must be"),
                              ({"start": "09:00", "end": "17:00", "days": [["mon"]]}, "hours.days must be")]],
    ({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "unless": {"tool": "t2"}}, "actions": []}]},
     "when.unless is only valid on time triggers"),
    ({"triggers": [{"id": "x", "when": {"type": "time", "every": "PT1H", "unless": {"tool": "t2"}}, "actions": []}]},
     "only valid on a one-shot time trigger"),
    ({"triggers": [{"id": "x", "when": {"type": "time", "at": "PT1H", "unless": {"where": {}}}, "actions": []}]},
     "must be an object with a non-empty 'tool'"),
    ({"triggers": [{"id": "x", "when": {"type": "time", "at": "PT1H", "unless": {"tool": "t2", "roles": []}},
                    "actions": []}]}, "when.unless.roles must be a non-empty list"),
    *[({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "roles": roles}, "actions": []}]},
       "when.roles must be a non-empty list") for roles in ("denise", [], [""], [5])],
    ({"triggers": [{"id": "x", "when": {"type": "time", "after": "y", "offset": "PT1S", "roles": ["denise"]},
                    "actions": []}]}, "only valid on action and state triggers"),
    ({"executors": {"denise": {"a2a_url": "http://d", "role": "denise@example.com"}},
      "triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "roles": ["denise@example.com"]},
                    "actions": [{"type": "nl", "instruction": "hi", "executor": "denise"}]}]},
     "would provoke it again"),
    ({"executor": {"a2a_url": "http://e", "role": "executor"},
      "triggers": [{"id": "x", "when": {"type": "state", "roles": ["executor"], "check": {
          "tool": "t", "predicate": {"exists": True}}}, "actions": [{"type": "nl", "instruction": "hi"}]}]},
     "would provoke it again"),
])
def test_registration_fails_loud(engine, bad, fragment):
    with pytest.raises(TriggerError, match=".*"):
        try:
            engine.register(bad)
        except TriggerError as e:
            assert fragment in str(e)
            raise


@pytest.mark.asyncio
async def test_executor_role_never_watched_even_if_forced(engine):
    # Defense-in-depth: even if watch_roles contained the executor role, the structural skip means
    # its calls are never evaluated (registration normally rejects this; we set state directly to prove it).
    engine.register({"executor": {"a2a_url": "http://e", "role": "executor"},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "search_emails"},
                                   "actions": []}]})
    engine._watch_roles.add("executor")  # force the dangerous state past validation
    engine.on_tool_call("executor", "search_emails", {}, _result())
    await asyncio.sleep(0.1)
    assert engine._triggers["x"]["status"] == "armed", "executor traffic must never fire a trigger"


def _trig(which):
    return _registration()["triggers"][which]


def test_additive_and_remove(engine):
    r1 = engine.register({"watch_roles": ["default"], "triggers": [_trig(0)]})  # grant
    assert r1["ok"] and r1["added"] == ["grant"] and r1["all"] == ["grant"]
    r2 = engine.register({"triggers": [_trig(2)]})  # sensor; watch_roles inherited
    assert r2["added"] == ["sensor"] and set(r2["all"]) == {"grant", "sensor"}
    # additive, NOT replace-all: both survive
    assert {t["id"] for t in engine.state()["triggers"]} == {"grant", "sensor"}
    # remove one
    assert engine.remove({"ids": ["grant"]})["removed"] == ["grant"]
    assert {t["id"] for t in engine.state()["triggers"]} == {"sensor"}
    # unknown-id remove is a no-op
    assert engine.remove({"ids": ["nope"]})["removed"] == []
    # a removed id can be re-added (no retirement)
    r3 = engine.register({"triggers": [_trig(0)]})
    assert r3["added"] == ["grant"] and set(r3["all"]) == {"grant", "sensor"}


def test_idempotent_readd_is_noop(engine):
    engine.register({"watch_roles": ["default"], "triggers": [_trig(0)]})
    engine.register({"triggers": [_trig(0)]})  # identical re-add
    assert [t["id"] for t in engine.state()["triggers"]] == ["grant"]


def test_conflicting_readd_rejected(engine):
    engine.register({"watch_roles": ["default"], "triggers": [_trig(0)]})
    conflicting = {"id": "grant", "when": {"type": "action", "tool": "different_tool"}, "actions": []}
    with pytest.raises(TriggerError, match="different spec"):
        engine.register({"triggers": [conflicting]})


@pytest.mark.asyncio
async def test_fired_trigger_stays_fired_on_readd(engine):
    engine.register(_registration())
    engine.on_tool_call("default", "gdocs_create_document", {"title": "Probe Doc"}, _result())
    await asyncio.sleep(0.2)
    assert engine._triggers["grant"]["status"] == "fired"
    engine.register({"triggers": [_trig(0)]})  # identical re-add must NOT re-arm
    assert engine._triggers["grant"]["status"] == "fired"


def test_clear_full_reset(engine):
    engine.register(_registration())
    assert engine.state()["triggers"]
    engine.clear()
    st = engine.state()
    assert st["triggers"] == [] and st["events"] == []
    assert st["config"] == {"watch_roles": [], "executor_configured": False, "executors": []}
    # after clear, config is settable fresh (different watch_roles allowed)
    engine.register({"watch_roles": ["solver"], "triggers": []})
    assert engine.state()["config"]["watch_roles"] == ["solver"]


def test_config_set_once(engine):
    engine.register({"watch_roles": ["default"], "triggers": []})
    engine.register({"watch_roles": ["default"], "triggers": []})  # identical repeat: ok
    engine.register({"triggers": []})  # omit: inherit
    assert engine.state()["config"]["watch_roles"] == ["default"]
    with pytest.raises(TriggerError, match="conflicts with the already-registered"):
        engine.register({"watch_roles": ["default", "other"], "triggers": []})


def test_executor_set_once_and_unset_to_set(engine):
    engine.register({"watch_roles": ["default"], "triggers": []})
    assert engine.state()["config"]["executor_configured"] is False
    engine.register({"executor": {"a2a_url": "http://e", "role": "executor"}, "triggers": []})  # unset -> set
    assert engine.state()["config"]["executor_configured"] is True
    with pytest.raises(TriggerError, match="executor conflicts"):
        engine.register({"executor": {"a2a_url": "http://other", "role": "executor"}, "triggers": []})


def test_named_executors_are_additive_and_set_once_per_name(engine):
    alice = {"a2a_url": "http://alice/", "role": "alice@example.com"}
    engine.register({"executors": {"alice": alice}, "triggers": []})
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob@example.com"}}, "triggers": []})
    engine.register({"executors": {"alice": {**alice, "timeout_seconds": 120}}, "triggers": []})  # identical re-add
    assert engine.state()["config"]["executors"] == ["alice", "bob"]
    assert engine.state()["config"]["executor_configured"] is False
    with pytest.raises(TriggerError, match="executor 'alice' conflicts"):
        engine.register({"executors": {"alice": {**alice, "a2a_url": "http://elsewhere"}}, "triggers": []})
    engine.clear()
    assert engine.state()["config"]["executors"] == []


def test_a_named_executor_registered_earlier_serves_a_later_batch(engine):
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}}, "triggers": []})
    added = engine.register({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                           "actions": [{"type": "nl", "instruction": "hi", "executor": "bob"}]}]})
    assert added["added"] == ["x"]


@pytest.mark.asyncio
async def test_each_nl_action_goes_to_the_executor_it_names(engine, executor_calls):
    engine.register({
        "executor": {"a2a_url": "http://default", "role": "executor"},
        "executors": {"alice": {"a2a_url": "http://alice", "role": "alice@example.com"},
                      "bob": {"a2a_url": "http://bob", "role": "bob@example.com"}},
        "triggers": [{"id": tid, "when": {"type": "action", "tool": "linear_create_issue"},
                      "actions": [{"type": "nl", "instruction": f"as {tid}", **named}]}
                     for tid, named in (("d", {}), ("a", {"executor": "alice"}), ("b", {"executor": "bob"}))],
    })
    engine.on_tool_call("default", "linear_create_issue", {}, _result())
    await asyncio.sleep(0.2)

    assert sorted((e["a2a_url"], i, c) for e, i, c in executor_calls) == [
        ("http://alice", "as a", "trigger-a"), ("http://bob", "as b", "trigger-b"), ("http://default", "as d", "trigger-d")]
    details = {e["trigger_id"]: e["detail"] for e in engine._events if e["kind"] == "action_ok"}
    assert details == {"d": {"nl": "as d"}, "a": {"nl": "as a", "executor": "alice"},
                       "b": {"nl": "as b", "executor": "bob"}}


@pytest.mark.asyncio
async def test_a_named_executor_that_does_not_complete_is_named_in_the_failure(engine, executor_calls):
    executor_calls.reply = None
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "nl", "instruction": "hi", "executor": "bob"}]}]})
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.2)

    [failure] = [e for e in engine._events if e["kind"] == "action_failed"]
    assert failure["executor"] == "bob" and engine._triggers["x"]["status"] == "armed"


@pytest.mark.asyncio
async def test_a_named_executor_that_cannot_be_reached_is_named_in_the_failure(engine, executor_calls):
    executor_calls.reply = httpx.ConnectTimeout("")
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "nl", "instruction": "hi", "executor": "bob"}]}]})
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.2)

    [failure] = [e for e in engine._events if e["kind"] == "action_failed"]
    assert (failure["executor"], failure["detail"]) == ("bob", "ConnectTimeout")


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["executor", "alice@example.com", "bob@example.com"])
async def test_no_executor_role_is_ever_watched(engine, role):
    engine.register({"executor": {"a2a_url": "http://e", "role": "executor"},
                     "executors": {"alice": {"a2a_url": "http://a", "role": "alice@example.com"},
                                   "bob": {"a2a_url": "http://b", "role": "bob@example.com"}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "search_emails"}, "actions": []}]})
    engine._watch_roles.add(role)  # force the dangerous state past validation
    engine.on_tool_call(role, "search_emails", {}, _result())
    await asyncio.sleep(0.1)
    assert engine._triggers["x"]["status"] == "armed"


@pytest.mark.asyncio
async def test_a_trigger_listing_an_executor_role_is_provoked_by_that_executor_alone(engine, executor_calls):
    engine.register({
        "executors": {"denise": {"a2a_url": "http://d", "role": "denise@example.com"},
                      "bob": {"a2a_url": "http://b", "role": "bob@example.com"}},
        "triggers": [
            {"id": "denise-hears-solver", "when": {"type": "action", "tool": "slack_send_message", "repeat": True},
             "actions": [{"type": "nl", "executor": "denise", "instruction": "reply to the solver"}]},
            {"id": "bob-hears-denise", "when": {"type": "action", "tool": "slack_send_message", "repeat": True,
                                                "roles": ["denise@example.com"]},
             "actions": [{"type": "nl", "executor": "bob", "instruction": "reply to denise"}]},
        ],
    })
    for role in ("denise@example.com", "bob@example.com", "default"):
        engine.on_tool_call(role, "slack_send_message", {}, _result())
        await asyncio.sleep(0.2)

    assert [(e["role"], i) for e, i, _ in executor_calls] == [
        ("bob@example.com", "reply to denise"), ("denise@example.com", "reply to the solver")]
    detected = [(e["trigger_id"], e["provoking"]["role"]) for e in engine._events if e["kind"] == "detected"]
    assert detected == [("bob-hears-denise", "denise@example.com"), ("denise-hears-solver", "default")]
    assert {t["id"]: t["when"].get("roles") for t in engine.state()["triggers"]} == {
        "denise-hears-solver": None, "bob-hears-denise": ["denise@example.com"]}


@pytest.mark.asyncio
async def test_an_executor_write_prompts_only_the_state_triggers_that_list_it(engine):
    engine._test_state["doc_text"] = "Budget Fit"
    check = {"tool": "gdocs_get_document_text", "args": {"documentId": "doc-1"},
             "extract": "text", "predicate": {"regex": "Budget Fit"}}
    engine.register({"executors": {"denise": {"a2a_url": "http://d", "role": "denise@example.com"}},
                     "triggers": [{"id": "listed", "when": {"type": "state", "roles": ["denise@example.com"],
                                                            "check": check}, "actions": []},
                                  {"id": "unlisted", "when": {"type": "state", "check": check}, "actions": []}]})
    engine.on_tool_call("denise@example.com", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.2)
    assert {tid: t["status"] for tid, t in engine._triggers.items()} == {"listed": "fired", "unlisted": "armed"}


@pytest.mark.asyncio
async def test_a_delay_waits_its_virtual_time_before_the_action(engine, executor_calls, virtual_time):
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "reply",
                                                "delay": "PT20M"}]}]})
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.2)

    assert virtual_time == [1200.0] and len(executor_calls) == 1
    [delayed] = [e for e in engine._events if e["kind"] == "delayed"]
    [acted] = [e for e in engine._events if e["kind"] == "action_ok"]
    assert (delayed["virtual_seconds"], delayed["executor"]) == (1200.0, "bob")
    assert (delayed["virtual_time"], acted["virtual_time"]) == ("2026-10-14T08:00:00Z", "2026-10-14T08:20:00Z")


@pytest.mark.asyncio
async def test_queued_calls_each_wait_out_their_delay_from_their_own_call(engine, virtual_time):
    engine.register({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "repeat": True},
                                   "actions": [{"type": "tool", "tool": "slack_send_message",
                                                "args": {"channel": "#c", "text": "hi"}, "delay": "PT1H"}]}]})
    for _ in range(3):  # 08:00, 08:01 and 08:02, all before the first reply's turn
        engine.on_tool_call("default", "t", {}, _result())
        virtual_time.advance(60)
    await asyncio.sleep(0.2)

    replies = [e["virtual_time"] for e in engine._events if e["kind"] == "action_ok"]
    assert replies == ["2026-10-14T09:00:00Z", "2026-10-14T09:01:00Z", "2026-10-14T09:02:00Z"]


@pytest.mark.asyncio
async def test_due_marks_wait_their_delay_from_the_mark_not_from_the_last_reply(engine, virtual_time):
    engine.register({"triggers": [{"id": "x", "when": {"type": "time", "every": "PT1H"},
                                   "actions": [{"type": "tool", "tool": "slack_send_message",
                                                "args": {"channel": "#c", "text": "hi"}, "delay": "PT30M"}]}]})
    engine._eval_time_triggers()
    virtual_time.advance(3 * 3600)  # 11:00: the 09:00, 10:00 and 11:00 marks come due together
    engine._eval_time_triggers()
    await asyncio.sleep(0.2)

    replies = [e["virtual_time"] for e in engine._events if e["kind"] == "action_ok"]
    assert replies == ["2026-10-14T11:00:00Z", "2026-10-14T11:00:00Z", "2026-10-14T11:30:00Z"]


@pytest.mark.asyncio
async def test_a_stochastic_delay_draws_the_same_sequence_from_its_seed(engine, waits):
    engine.register({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "repeat": True},
                                   "actions": [{"type": "tool", "tool": "slack_send_message",
                                                "args": {"channel": "#c", "text": "hi"},
                                                "delay": {"dist": "exp", "mean": "PT10M", "seed": 7}}]}]})
    for _ in range(3):
        engine.on_tool_call("default", "t", {}, _result())
        await asyncio.sleep(0.1)

    rng = random.Random(7)
    assert waits == [rng.expovariate(1.0 / 600) for _ in range(3)]
    assert len(engine._test_state["slack"]) == 3


@pytest.mark.asyncio
async def test_a_re_arm_during_one_mark_s_delay_does_not_shorten_the_next(engine, virtual_time, monkeypatch):
    engine.register({"triggers": [{"id": "x", "when": {"type": "time", "every": "PT1H"},
                                   "actions": [{"type": "tool", "tool": "slack_send_message",
                                                "args": {"channel": "#c", "text": "hi"}, "delay": "PT90M"}]}]})
    record = TriggerEngine._sleep_virtual

    async def re_arm_a_day_back_after_the_first_wait(self, seconds, generation=None):
        await record(self, seconds, generation)
        if len(virtual_time) == 1:
            self._gw._clock.set_time("2026-10-13T10:30:00Z")

    monkeypatch.setattr(TriggerEngine, "_sleep_virtual", re_arm_a_day_back_after_the_first_wait)
    engine._eval_time_triggers()
    virtual_time.advance(2 * 3600)  # 10:00: the 09:00 and 10:00 marks come due together
    engine._eval_time_triggers()
    await asyncio.sleep(0.2)

    assert virtual_time == [1800.0, 3600.0]


@pytest.mark.asyncio
async def test_working_hours_defer_an_nl_action_until_they_open(engine, executor_calls, virtual_time):
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob",
                                           "hours": {"start": "09:00", "end": "17:00"}}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "reply"}]}]})
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.2)

    [deferred] = [e for e in engine._events if e["kind"] == "deferred"]
    [acted] = [e for e in engine._events if e["kind"] == "action_ok"]
    assert (deferred["until"], deferred["executor"], acted["virtual_time"]) == (
        "2026-10-14T09:00:00Z", "bob", "2026-10-14T09:00:00Z")
    assert virtual_time == [3600.0] and len(executor_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change, until", [
    (lambda c: c.set_time("2026-10-19T10:00:00Z", rate=3600), ["2026-10-19T09:00:00Z"]),
    (lambda c: c.set_time("2026-10-17T12:00:00Z", rate=3600), ["2026-10-19T09:00:00Z"] * 2),
    (lambda c: c.clear(), ["2026-10-19T09:00:00Z"]),
], ids=["re-armed-into-hours", "re-armed-still-closed", "cleared"])
async def test_working_hours_are_reckoned_again_after_a_re_arm_and_lifted_by_a_clear(engine, monkeypatch, change,
                                                                                      until):
    stepped = _SteppedClock(engine, monkeypatch)
    stepped.clock.set_time("2026-10-16T18:00:00Z", rate=3600)  # a Friday evening, closed until Monday 09:00
    stepped.at[1] = lambda: change(stepped.clock)
    trig = {"waiting": False}
    hours = {"start": "09:00", "end": "17:00", "utc_offset": "+00:00", "days": ["mon", "tue", "wed", "thu", "fri"]}

    await engine._await_hours("x", trig, 0, hours, {})

    assert [e["until"] for e in engine._events if e["kind"] == "deferred"] == until and not trig["waiting"]
    now = stepped.clock.now()
    assert now is None or now.strftime("%a %H:%M") in ("Mon 09:00", "Mon 10:00")


@pytest.mark.asyncio
async def test_working_hours_do_not_apply_while_the_clock_is_unarmed(engine, executor_calls, waits):
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob",
                                           "hours": {"start": "09:00", "end": "09:01"}}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "reply"}]}]})
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.2)

    assert waits == [] and len(executor_calls) == 1
    assert not [e for e in engine._events if e["kind"] == "deferred"]


@pytest.mark.parametrize("now, hours, opens", [
    ("2026-10-14T08:00:00Z", {"start": "09:00", "end": "17:00"}, "2026-10-14T09:00:00Z"),
    ("2026-10-14T12:00:00Z", {"start": "09:00", "end": "17:00"}, None),
    ("2026-10-14T17:00:00Z", {"start": "09:00", "end": "17:00"}, "2026-10-15T09:00:00Z"),
    ("2026-10-14T15:00:00Z", {"start": "09:00", "end": "17:00", "utc_offset": "-07:00"}, "2026-10-14T16:00:00Z"),
    ("2026-10-14T16:30:00Z", {"start": "09:00", "end": "17:00", "utc_offset": "-07:00"}, None),
    ("2026-10-14T23:00:00Z", {"start": "22:00", "end": "06:00"}, None),
    ("2026-10-15T03:00:00Z", {"start": "22:00", "end": "06:00"}, None),
    ("2026-10-14T12:00:00Z", {"start": "22:00", "end": "06:00"}, "2026-10-14T22:00:00Z"),
    ("2026-10-17T12:00:00Z", {"start": "09:00", "end": "17:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
     "2026-10-19T09:00:00Z"),
    ("2026-10-17T03:00:00Z", {"start": "22:00", "end": "06:00", "days": ["fri"]}, None),
], ids=["before", "inside", "at-end", "offset-before", "offset-inside", "wrap-evening", "wrap-morning",
        "wrap-midday", "weekend", "wrap-from-a-listed-day"])
def test_next_opening(now, hours, opens):
    hours = {"utc_offset": "+00:00", "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"], **hours}
    got = _next_opening(datetime.fromisoformat(now.replace("Z", "+00:00")), hours)
    assert (got.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if got else None) == opens


def _deadline(**unless) -> dict:
    """An anchor `dm` (fired by a call to `t`) and a deadline 30 virtual minutes after it, cancelled by `unless`."""
    return {"executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
            "triggers": [{"id": "dm", "when": {"type": "action", "tool": "t"}, "actions": []},
                         {"id": "deadline", "when": {"type": "time", "after": "dm", "offset": "PT30M",
                                                     "unless": {"tool": "slack_send_message", **unless}},
                          "actions": []}]}


@pytest.mark.asyncio
async def test_a_matching_call_after_the_anchor_cancels_the_deadline(engine, virtual_time):
    engine.register(_deadline(where={"channel_id": {"equals": "D1"}}))
    engine.on_tool_call("default", "slack_send_message", {"channel_id": "D1"}, _result())  # before the anchor
    assert engine._triggers["deadline"]["status"] == "armed"
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.1)
    for call, error in ((("default", "slack_send_message", {"channel_id": "D2"}), False),
                        (("default", "slack_send_message", {"channel_id": "D1"}), True),
                        (("bob", "slack_send_message", {"channel_id": "D1"}), False)):
        engine.on_tool_call(*call, _result(is_error=error))
    assert engine._triggers["deadline"]["status"] == "armed"

    engine.on_tool_call("default", "slack_send_message", {"channel_id": "D1"}, _result())
    assert engine.state()["triggers"][1]["status"] == "cancelled"
    [cancelled] = [e for e in engine._events if e["kind"] == "cancelled"]
    assert cancelled["provoking"] == {"tool": "slack_send_message", "role": "default"}

    virtual_time.advance(3600)
    engine._eval_time_triggers()
    await asyncio.sleep(0.1)
    assert engine._triggers["deadline"]["status"] == "cancelled" and engine._triggers["deadline"]["fire_count"] == 0


@pytest.mark.asyncio
async def test_a_deadline_lists_the_executor_roles_that_cancel_it(engine, virtual_time):
    engine.register(_deadline(roles=["bob"]))
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.1)
    engine.on_tool_call("default", "slack_send_message", {}, _result())
    assert engine._triggers["deadline"]["status"] == "armed"
    engine.on_tool_call("bob", "slack_send_message", {}, _result())
    assert engine._triggers["deadline"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_a_deadline_nobody_cancels_fires_when_due(engine, virtual_time):
    engine.register(_deadline())
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.1)
    virtual_time.advance(29 * 60)
    engine._eval_time_triggers()
    assert engine._triggers["deadline"]["status"] == "armed"
    virtual_time.advance(2 * 60)
    engine._eval_time_triggers()
    await asyncio.sleep(0.1)
    assert engine._triggers["deadline"]["status"] == "fired"


@pytest.mark.asyncio
async def test_a_reply_while_the_anchor_is_still_acting_cancels_the_deadline(engine, monkeypatch, virtual_time):
    async def bob_dms_and_is_answered_before_he_reports_back(self, executor, instruction, context_id):
        engine.on_tool_call("default", "slack_send_message", {}, _result())
        return "done"

    monkeypatch.setattr(TriggerEngine, "_call_executor", bob_dms_and_is_answered_before_he_reports_back)
    registration = _deadline()
    registration["triggers"][0]["actions"] = [{"type": "nl", "executor": "bob", "instruction": "DM the solver"}]
    engine.register(registration)
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.1)
    assert (engine._triggers["dm"]["status"], engine._triggers["deadline"]["status"]) == ("fired", "cancelled")
    assert not [e for e in engine._events if e["kind"] == "anchored"]

    virtual_time.advance(3600)
    engine._eval_time_triggers()
    await asyncio.sleep(0.1)
    assert engine._triggers["deadline"]["fire_count"] == 0


@pytest.mark.asyncio
async def test_a_reply_after_the_deadline_is_due_comes_too_late(engine, virtual_time):
    engine.register(_deadline())
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.1)
    virtual_time.advance(31 * 60)  # due, though nothing has looked yet
    engine.on_tool_call("default", "slack_send_message", {}, _result())
    await asyncio.sleep(0.1)
    assert engine._triggers["deadline"]["status"] == "fired"
    assert not [e for e in engine._events if e["kind"] == "cancelled"]


@pytest.mark.asyncio
async def test_cancelling_a_deadline_retires_the_triggers_anchored_on_it(engine, virtual_time):
    engine.register({"triggers": [
        {"id": "nudge", "when": {"type": "time", "at": "PT1H", "unless": {"tool": "slack_send_message"}}, "actions": []},
        {"id": "escalate", "when": {"type": "time", "after": "nudge", "offset": "PT1H"}, "actions": []}]})
    engine.on_tool_call("default", "slack_send_message", {}, _result())

    assert {t["id"]: t["status"] for t in engine.state()["triggers"]} == {"nudge": "cancelled", "escalate": "cancelled"}
    assert not engine._has_time_triggers
    cancelled = [e for e in engine._events if e["kind"] == "cancelled"]
    assert [(e["trigger_id"], e.get("anchor")) for e in cancelled] == [("nudge", None), ("escalate", "nudge")]


def test_a_null_unless_means_none(engine):
    engine.register({"triggers": [{"id": "a", "when": {"type": "action", "tool": "t", "unless": None}, "actions": []},
                                  {"id": "b", "when": {"type": "time", "at": "PT1H", "unless": None}, "actions": []}]})
    assert all("unless" not in t["spec"]["when"] for t in engine._triggers.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", ["remove", "clear"])
async def test_removing_or_clearing_a_trigger_mid_delay_ends_its_fire_without_a_trace(engine, reset):
    engine.register({"triggers": [{"id": "x", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "tool", "tool": "slack_send_message",
                                                "args": {"channel": "#c", "text": "hi"}, "delay": "PT1H"}]}]})
    engine.on_tool_call("default", "t", {}, _result())
    await asyncio.sleep(0.05)
    assert engine.state()["triggers"][0]["waiting"] is True

    engine.remove({"ids": ["x"]}) if reset == "remove" else engine.clear()
    await asyncio.sleep(0.05)
    assert engine._test_state["slack"] == []
    expected = ["added", "detected", "delayed", "removed"] if reset == "remove" else []
    assert [e["kind"] for e in engine._events] == expected


@pytest.mark.asyncio
async def test_each_fire_is_numbered_failed_ones_included(engine, monkeypatch):
    seen, replies = [], iter([None, "done", "done"])

    async def executor(self, executor, instruction, context_id):
        seen.append(engine.caused_by("bob"))
        return next(replies)

    monkeypatch.setattr(TriggerEngine, "_call_executor", executor)
    engine.register({"executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
                     "triggers": [{"id": "x", "when": {"type": "action", "tool": "t", "repeat": True},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "reply"}]}]})
    for _ in range(3):
        engine.on_tool_call("default", "t", {}, _result())
        await asyncio.sleep(0.05)
    assert [causes[0]["fire"] for causes in seen] == [1, 2, 3] and engine._triggers["x"]["fire_count"] == 2


def test_generated_id_on_omit(engine):
    r = engine.register({"watch_roles": ["default"], "triggers": [
        {"when": {"type": "action", "tool": "t"}, "actions": []}]})
    assert len(r["added"]) == 1 and r["added"][0].startswith("trg_")
    assert {t["id"] for t in engine.state()["triggers"]} == set(r["added"])


@pytest.mark.asyncio
async def test_watch_roles_and_success_only_filtering(engine):
    engine.register(_registration())
    engine.on_tool_call("executor", "gdocs_create_document", {"title": "Probe Doc"}, _result())
    engine.on_tool_call("default", "gdocs_create_document", {"title": "Probe Doc"}, _result(is_error=True))
    engine.on_tool_call("default", "gdocs_create_document", {"title": "Some Other Doc"}, _result())
    await asyncio.sleep(0.1)
    assert engine._triggers["grant"]["status"] == "armed"


@pytest.mark.asyncio
async def test_action_trigger_fires_permission(engine):
    engine.register(_registration())
    engine.on_tool_call("default", "gdocs_create_document", {"title": "Probe Doc"}, _result())
    await asyncio.sleep(0.2)
    assert engine._triggers["grant"]["status"] == "fired"
    assert ("default", "snowflake_x", False) in engine._test_gw.rules
    kinds = [e["kind"] for e in engine._events]
    assert kinds.count("detected") == 1 and "fired" in kinds


@pytest.mark.asyncio
async def test_state_trigger_pipeline_tool_action_and_verify(engine):
    engine.register(_registration())
    engine.on_tool_call("default", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.3)
    assert engine._triggers["rate"]["status"] == "armed"  # no Budget Fit yet

    engine._test_state["doc_text"] = "Budget Fit: fine."
    engine.on_tool_call("default", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.3)
    assert engine._triggers["rate"]["status"] == "fired"
    assert engine._test_state["slack"] and "$26/hr" in engine._test_state["slack"][0]["text"]
    assert any(e["kind"] == "verify_ok" for e in engine._events)


@pytest.mark.asyncio
async def test_a_tool_action_s_as_role_reaches_its_call_and_verify_but_not_the_state_check(engine):
    reg = _registration()
    rate = reg["triggers"][1]
    rate["actions"][0]["as"] = "carol@example.com"
    rate["actions"].append({"type": "tool", "tool": "slack_send_message", "args": {"channel": "#c", "text": "plain"}})
    engine.register(reg)
    engine._test_state["doc_text"] = "Budget Fit: fine."
    engine.on_tool_call("default", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.3)
    assert engine._triggers["rate"]["status"] == "fired"
    gw = engine._test_gw
    assert all(role is None for tool, role in gw.roles if tool.startswith("gdocs_"))  # when.check reads as the pin
    assert [role for tool, role in gw.roles if tool == "slack_send_message"] == ["carol@example.com", None]
    assert [role for tool, role in gw.roles if tool == "slack_conversations_history"] == ["carol@example.com"]
    oks = [e for e in engine._events if e["kind"] == "action_ok"]
    assert [e["detail"].get("as") for e in oks] == ["carol@example.com", None]
    assert any(e["kind"] == "verify_ok" for e in engine._events)
    internal = [e for e in gw.events if e["event_type"] == "internal_tool_call"]
    assert [(e["as"], e["ok"]) for e in internal] == [("carol@example.com", True), (None, True)]


@pytest.mark.asyncio
async def test_a_tool_action_s_as_role_is_stripped_once_at_registration(engine):
    reg = _registration()
    reg["triggers"][1]["actions"][0]["as"] = " carol@example.com "
    engine.register(reg)
    assert engine._triggers["rate"]["spec"]["actions"][0]["as"] == "carol@example.com"
    engine._test_state["doc_text"] = "Budget Fit: fine."
    engine.on_tool_call("default", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.3)
    assert engine._triggers["rate"]["status"] == "fired"
    assert [role for tool, role in engine._test_gw.roles if tool.startswith("slack_")] == ["carol@example.com"] * 2
    assert [e["detail"]["as"] for e in engine._events if e["kind"] == "action_ok"] == ["carol@example.com"]


@pytest.mark.asyncio
async def test_cancel_scheduled_cancels_every_pending_action_and_waits_for_it(engine):
    async def pending():
        await asyncio.sleep(60)

    tasks = [engine._schedule(pending()) for _ in range(2)]
    await asyncio.sleep(0)
    await engine.cancel_scheduled()
    assert all(t.cancelled() for t in tasks) and engine._tasks == set()


@pytest.mark.asyncio
async def test_sensor_trigger_and_state_shape(engine):
    engine.register(_registration())
    engine._test_state["doc_text"] = "rate is $26/hr now"
    engine.on_tool_call("default", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.3)
    assert engine._triggers["sensor"]["status"] == "fired"
    state = engine.state()
    assert "sensor" in {t["id"] for t in state["triggers"]}
    assert state["config"] == {"watch_roles": ["default"], "executor_configured": False, "executors": []}
    assert state["events"][0]["kind"] == "added"
    assert all("seq" in e and "ts" in e for e in state["events"])


@pytest.mark.asyncio
async def test_nl_action_via_stub_executor_with_verify(engine):
    import threading
    import uuid as _uuid
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pytest_socket import enable_socket

    enable_socket()  # local loopback stub executor; re-disabled by the autouse fixture's next setup

    tasks: dict = {}

    class _Exec(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req["method"] == "message/send":
                tid = _uuid.uuid4().hex[:8]
                text = req["params"]["message"]["parts"][0]["text"]
                if "NLMARK-77" in text:
                    engine._test_state["slack"].append({"text": "executor says NLMARK-77"})
                tasks[tid] = "completed"
                body = {"jsonrpc": "2.0", "id": req["id"], "result": {"id": tid, "status": {"state": "submitted"}}}
            else:
                body = {"jsonrpc": "2.0", "id": req["id"],
                        "result": {"id": req["params"]["id"],
                                   "status": {"state": tasks[req["params"]["id"]],
                                              "message": {"parts": [{"kind": "text", "text": "DONE"}]}}}}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Exec)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        engine.register({
            "executor": {"a2a_url": f"http://127.0.0.1:{server.server_port}", "timeout_seconds": 15,
                         "role": "executor"},
            "triggers": [{"id": "nl", "when": {"type": "action", "tool": "linear_create_issue"},
                          "actions": [{"type": "nl", "instruction": "post exactly NLMARK-77 in #c",
                                       "verify": {"tool": "slack_conversations_history", "args": {"channel_id": "c"},
                                                  "predicate": {"regex": "NLMARK-77"}}}]}],
        })
        engine.on_tool_call("default", "linear_create_issue", {"title": "x"}, _result())
        for _ in range(60):
            await asyncio.sleep(0.1)
            if engine._triggers["nl"]["status"] != "firing" and engine._triggers["nl"]["detected_at"]:
                break
        assert engine._triggers["nl"]["status"] == "fired", engine.state()["events"][-3:]
        assert any(e["kind"] == "verify_ok" for e in engine._events)
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_broken_check_is_fail_open(engine):
    engine.register({"triggers": [
        {"id": "broken", "when": {"type": "state", "check": {"steps": [
            {"tool": "nonexistent_tool", "args": {}, "predicate": {"exists": True}}]}},
         "actions": []},
    ]})
    engine.on_tool_call("default", "any_write_tool", {}, _result())
    await asyncio.sleep(0.2)
    assert engine._triggers["broken"]["status"] == "armed"
    assert any(e["kind"] == "eval_error" for e in engine._events)


# --- Gateway-level: the /step REST path fires triggers too ---


def _step_gateway():
    from agent_env.env.gateway.gateway import Gateway

    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway",
                 internal_mcp_servers=[], rest_proxy_urls={})
    gw._tool_server_urls = {"gdocs_create_document": "http://svc/mcp"}

    async def _fake_call_child_tool(server_url, tool_name, arguments, *, caller=None):
        return _result()

    gw._call_child_tool = _fake_call_child_tool  # type: ignore[assignment]
    return gw


@pytest.mark.asyncio
async def test_step_path_fires_triggers(monkeypatch):
    gw = _step_gateway()
    gw._trigger_engine.register({
        "watch_roles": ["default"],
        "triggers": [{"id": "s", "when": {"type": "action", "tool": "gdocs_create_document",
                                          "where": {"title": {"regex": "Probe"}}},
                      "actions": []}],
    })
    resp = await gw._step_call_tool({"tool_name": "gdocs_create_document", "arguments": {"title": "Probe Doc"}},
                                    "default")
    assert resp.status_code == 200
    await asyncio.sleep(0.1)
    assert gw._trigger_engine._triggers["s"]["status"] == "fired"
    ev = gw._trigger_engine.state()["events"]
    assert any(e["kind"] == "detected" and e.get("provoking", {}).get("tool") == "gdocs_create_document" for e in ev)


@pytest.mark.asyncio
async def test_step_path_respects_watch_roles(monkeypatch):
    gw = _step_gateway()
    gw._trigger_engine.register({
        "watch_roles": ["default"],
        "triggers": [{"id": "s", "when": {"type": "action", "tool": "gdocs_create_document"},
                      "actions": []}],
    })
    resp = await gw._step_call_tool({"tool_name": "gdocs_create_document", "arguments": {"title": "Probe Doc"}},
                                    "harness")
    assert resp.status_code == 200
    await asyncio.sleep(0.1)
    assert gw._trigger_engine._triggers["s"]["status"] == "armed"


class _ExecutorCalls(list):
    reply: object = "done"



@pytest.mark.asyncio
@pytest.mark.parametrize("rate, virtual_seconds, steps", [(60, 1200, [5.0, 5.0, 5.0, 5.0]), (None, 12, [5.0, 5.0, 2.0])],
                         ids=["armed-at-60x", "unarmed"])
async def test_waiting_virtual_time_follows_the_clock_rate(engine, monkeypatch, rate, virtual_seconds, steps):
    stepped = _SteppedClock(engine, monkeypatch)
    if rate is not None:
        stepped.clock.set_time("2026-10-14T08:00:00Z", rate=rate)
    await engine._sleep_virtual(virtual_seconds)
    assert stepped.slept == steps


@pytest.mark.asyncio
async def test_waiting_virtual_time_holds_while_frozen_and_follows_a_new_rate(engine, monkeypatch):
    stepped = _SteppedClock(engine, monkeypatch)
    stepped.at[1] = lambda: stepped.clock.set_time("2026-10-14T08:00:00Z", rate=0)  # after 5 unarmed seconds
    stepped.at[3] = lambda: stepped.clock.set_time("2026-10-14T08:00:00Z", rate=4)  # where it froze, now at 4x
    await engine._sleep_virtual(13)
    assert stepped.slept == [5.0, 5.0, 5.0, 2.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("re_arm_to, steps", [("2026-10-14T08:00:08Z", [5.0, 2.0]), ("2026-10-13T08:00:00Z", [5.0, 5.0])],
                         ids=["forward-counts-its-jump", "back-counts-nothing"])
async def test_waiting_virtual_time_counts_a_re_arm_forward_but_not_back(engine, monkeypatch, re_arm_to, steps):
    stepped = _SteppedClock(engine, monkeypatch)
    stepped.clock.set_time("2026-10-14T08:00:00Z")
    stepped.at[1] = lambda: stepped.clock.set_time(re_arm_to)  # at 08:00:05: 3 s ahead, or a day back
    await engine._sleep_virtual(10)
    assert stepped.slept == steps


@pytest.mark.asyncio
async def test_a_wait_tied_to_a_clock_generation_ends_when_the_clock_moves_on(engine, monkeypatch):
    stepped = _SteppedClock(engine, monkeypatch)
    stepped.clock.set_time("2026-10-14T08:00:00Z")
    stepped.at[1] = stepped.clock.clear
    await engine._sleep_virtual(3600, generation=stepped.clock.generation)
    assert stepped.slept == [5.0]


@pytest.mark.asyncio
async def test_tool_call_events_carry_the_callers_role_and_the_fire_an_executor_made_them_for(monkeypatch):
    gw = _step_gateway()
    events: list[dict] = []

    async def capture(event):
        events.append(dict(event))
        return f"event_{len(events)}"

    async def executor(self, executor, instruction, context_id):
        await gw._step_call_tool({"tool_name": "gdocs_create_document", "arguments": {"title": "by bob"}}, "bob")
        return "done"

    monkeypatch.setattr(gw, "_log_event", capture)
    monkeypatch.setattr(TriggerEngine, "_call_executor", executor)
    gw._trigger_engine.register({
        "watch_roles": ["default"], "executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
        "triggers": [{"id": "t", "when": {"type": "action", "tool": "gdocs_create_document",
                                          "where": {"title": {"regex": "solver"}}},
                      "actions": [{"type": "nl", "executor": "bob", "instruction": "write"}]}]})
    await gw._step_call_tool({"tool_name": "gdocs_create_document", "arguments": {"title": "by solver"}}, "default")
    await asyncio.sleep(0.2)
    await gw._step_call_tool({"tool_name": "gdocs_create_document", "arguments": {"title": "later"}}, "bob")

    calls = [(e["tool_call"]["arguments"]["title"], e["role"], e.get("caused_by"))
             for e in events if e["event_type"] == "tool_call"]
    assert calls == [("by solver", "default", None),
                     ("by bob", "bob", [{"trigger_id": "t", "action_index": 0, "fire": 1}]),
                     ("later", "bob", None)]


class _Waits(list):
    advance = None


def _fake_waits(engine, monkeypatch, armed_at=None) -> _Waits:
    source = {"t": 0.0}
    clock = Clock(now_fn=lambda: source["t"])
    if armed_at is not None:
        clock.set_time(armed_at)
    engine._test_gw._clock = clock
    waited = _Waits()
    waited.advance = lambda seconds: source.__setitem__("t", source["t"] + seconds / clock.rate)

    async def fake_sleep(self, seconds, generation=None):
        waited.append(seconds)
        waited.advance(max(0.0, seconds))

    monkeypatch.setattr(TriggerEngine, "_sleep_virtual", fake_sleep)
    return waited


class _SteppedClock:
    """The engine's clock on a fake source that asyncio.sleep advances, so the real virtual-time waits run at once;
    `.at[n]` runs after the n-th sleep."""

    def __init__(self, engine, monkeypatch):
        self.t = 0.0
        self.slept: list[float] = []
        self.at: dict = {}
        self.clock = engine._test_gw._clock = Clock(now_fn=lambda: self.t)

        async def fake_sleep(seconds):
            self.slept.append(seconds)
            assert len(self.slept) < 100, "the wait never ended"
            self.t += seconds
            self.at.pop(len(self.slept), lambda: None)()

        monkeypatch.setattr(triggers_module.asyncio, "sleep", fake_sleep)
