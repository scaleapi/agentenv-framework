"""Unit tests for the provoking-call context additions to the trigger engine (urn:agentenv:triggers/v1):
`${args.*}` / `${result.*}` templating, `where` over the result, repeat + queue, failure re-arm,
barriers and registration-time tool checks."""
from __future__ import annotations

import asyncio
import json
import re

import pytest
from mcp.types import CallToolResult, TextContent

from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError, _template, TemplateError


class _Tool:
    def __init__(self, name):
        self.name = name


class _FakeGateway:
    def __init__(self, responses: dict, known_tools: list[str] | None = None) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []
        self.rules: list[tuple[str, str, bool]] = []
        self.events: list[dict] = []
        self._role_rules_lock = asyncio.Lock()
        self._server_tools: dict = {"svc": [_Tool(n) for n in known_tools]} if known_tools else {}
        self._tool_server_urls: dict = {}
        self._clock = Clock()
        self.TOOL_CALL_TIMEOUT_S = 30
        self.TRIGGER_BARRIER_TIMEOUT_S = 30.0
        self.changelog = 10

    def _apply_rule(self, role, tool, value):
        self.rules.append((role, tool, value))

    def _query_changelog_id(self):
        return self.changelog

    async def _log_event(self, event):
        self.events.append(dict(event))
        return f"event_{len(self.events)}"

    async def _ensure_tools_discovered(self):
        pass


def _result(text: str = "{}", is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


def _make_engine(monkeypatch, responses: dict, known_tools=None, delays: dict | None = None):
    gw = _FakeGateway(responses, known_tools)
    delays = delays or {}

    async def fake_internal_call(self, tool_name, arguments):
        gw.calls.append((tool_name, dict(arguments) if isinstance(arguments, dict) else arguments))
        if tool_name in delays:
            await asyncio.sleep(delays[tool_name])
        fn = gw.responses.get(tool_name)
        if fn is None:
            raise RuntimeError(f"unknown tool {tool_name}")
        out = fn(arguments)
        gw.changelog += 1
        if isinstance(out, CallToolResult):  # a stub that answers with isError set
            return out
        return _result(out if isinstance(out, str) else json.dumps(out))

    monkeypatch.setattr(TriggerEngine, "_internal_call", fake_internal_call)
    eng = TriggerEngine(gw)
    eng._test_gw = gw
    return eng


async def _settle(engine, timeout=2.0):
    """Wait until no trigger is firing/queued (bounded)."""
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        if all(t["status"] not in ("firing", "queued") for t in engine._triggers.values()):
            return
    raise AssertionError(f"engine did not settle: {[(k, v['status']) for k, v in engine._triggers.items()]}")


TOUCH = {"id": "touch", "when": {"type": "action", "tool": "gsheets_values_update", "repeat": True},
         "actions": [{"type": "tool", "tool": "gdrive_touch_file", "args": {"fileId": "${args.spreadsheetId}"}}]}

CREATE = {"id": "create", "when": {"type": "action", "tool": "gdocs_create_document", "repeat": True,
                                   "where": {"driveFileId": {"exists": False}}},
          "barrier": {"at": "provoking_call"},
          "actions": [{"type": "tool", "tool": "gdrive_register_native_file",
                       "args": {"id": "${result.documentId}", "name": "${args.title}",
                                "mimeType": "application/vnd.google-apps.document",
                                "ownerEmail": "${result.owners[0]}", "createdTime": "${result.createdTime}"},
                       "verify": {"tool": "gdrive_get_file_metadata", "args": {"fileId": "${result.documentId}"},
                                  "extract": "id", "predicate": {"exists": True}}}]}


# ---------------------------------------------------------------- templating (E2)

def test_template_resolves_args_result_paths_and_raw_values():
    ctx = {"args": {"title": "Budget", "tags": ["a", "b"]},
           "result": {"documentId": "d1", "owners": ["o@x.co"], "nested": {"n": 3}, "guests": ["g1", "g2"]}}
    out = _template({"id": "${result.documentId}", "name": "${args.title}", "owner": "${result.owners[0]}",
                     "to": "${result.guests}", "n": "${result.nested.n}", "label": "doc ${result.documentId} by ${args.title}",
                     "all": "${args}", "static": 7, "list": ["${args.title}", {"k": "${result.nested.n}"}]}, {}, ctx)
    assert out == {"id": "d1", "name": "Budget", "owner": "o@x.co", "to": ["g1", "g2"], "n": 3,
                   "label": "doc d1 by Budget", "all": {"title": "Budget", "tags": ["a", "b"]}, "static": 7,
                   "list": ["Budget", {"k": 3}]}


def test_template_unresolved_ctx_raises_and_bare_names_stay_literal():
    with pytest.raises(TemplateError):
        _template({"x": "${result.missing}"}, {}, {"args": {}, "result": {}})
    with pytest.raises(TemplateError):
        _template({"x": "id=${args.nope}"}, {}, {"args": {}, "result": None})
    # legacy: an unknown bare binding stays literal; a known binding wins over ctx
    assert _template({"x": "${doc_id}", "y": "${result.documentId}"}, {"doc_id": "b1"},
                     {"args": {}, "result": {"documentId": "r1"}}) == {"x": "b1", "y": "r1"}
    assert _template({"x": "${unknown}"}, {}, {"args": {}, "result": {}}) == {"x": "${unknown}"}
    # v1 kept: a whole-placeholder check binding stays a string; only args./result. pass raw values
    assert _template({"id": "${row}", "n": "${result.n}"}, {"row": 42}, {"args": {}, "result": {"n": 42}}) == {"id": "42", "n": 42}
    try:
        _template({"x": "${args.missing}"}, {}, {"args": {}, "result": {}})
    except TemplateError as e:
        assert str(e) == "unresolved placeholder ${args.missing}"


# ---------------------------------------------------------------- registration (E3, E10, E11)

@pytest.mark.parametrize("bad,fragment", [
    ({"id": "x", "when": {"type": "time", "at": "PT1H"},
      "actions": [{"type": "tool", "tool": "t", "args": {"a": "${args.b}"}}]}, "only action triggers carry"),
    ({"id": "x", "when": {"type": "action", "tool": "t", "repeat": "yes"}, "actions": []}, "repeat must be a boolean"),
    ({"id": "x", "when": {"type": "action", "tool": "t"}, "barrier": True, "actions": []},
     "barrier must be an object"),
    ({"id": "x", "when": {"type": "action", "tool": "t"}, "barrier": {}, "actions": []},
     "barrier.at must be one of"),
    ({"id": "x", "when": {"type": "action", "tool": "t"}, "barrier": {"at": "next_call"}, "actions": []},
     "barrier.at must be one of"),
    ({"id": "x", "when": {"type": "action", "tool": "t"},
      "barrier": {"at": "provoking_call", "timout_seconds": 5}, "actions": []},
     "unknown key(s) ['timout_seconds']"),
    ({"id": "x", "when": {"type": "action", "tool": "t"},
      "barrier": {"at": "provoking_call", "timeout_seconds": 0}, "actions": []},
     "timeout_seconds must be a positive, finite number"),
    ({"id": "x", "when": {"type": "time", "at": "PT1H"}, "barrier": {"at": "provoking_call"}, "actions": []},
     "only valid on action triggers"),
    ({"id": "x", "when": {"type": "action", "tool": "t"},
      "actions": [{"type": "tool", "tool": "t2", "args": {"a": "${bogus.path}"}}]}, "must be a check binding name"),
    ({"id": "x", "when": {"type": "action", "tool": "t", "where": {"result.a b": {"exists": True}}},
      "actions": []}, "invalid path segment"),
    ({"id": "x", "when": {"type": "state", "check": {"tool": "t", "predicate": {"exists": True}}, "repeat": True},
      "actions": []}, "only valid on action triggers"),
    ({"id": "x", "when": {"type": "state", "check": {"tool": "t", "args": {"q": "${args.x}"}, "predicate": {"exists": True}}},
      "actions": []}, "only action triggers carry"),
    ({"id": "x", "when": {"type": "action", "tool": "t"},
      "actions": [{"type": "tool", "tool": "t2", "args": {"a": "${result.}"}}]}, "has an empty path"),
])
def test_registration_validation(monkeypatch, bad, fragment):
    engine = _make_engine(monkeypatch, {})
    with pytest.raises(TriggerError) as e:
        engine.register({"triggers": [bad]})
    assert fragment in str(e.value)


def test_registration_defaults_are_normalized_and_idempotent(monkeypatch):
    engine = _make_engine(monkeypatch, {})
    engine.register({"triggers": [TOUCH]})
    spec = engine._triggers["touch"]["spec"]
    assert "barrier" not in spec and engine._triggers["touch"]["failure_count"] == 0
    engine.register({"triggers": [TOUCH]})  # identical re-add: no-op
    st = engine.state()["triggers"][0]
    assert st["when"]["repeat"] is True and st["barrier"] is None and st["pending"] == 0
    assert st["failure_count"] == 0 and st["last_failure_at"] is None


def test_unknown_tools_rejected_at_registration_when_discovered(monkeypatch):
    engine = _make_engine(monkeypatch, {}, known_tools=["gsheets_values_update", "gdrive_touch_file"])
    engine.register({"triggers": [TOUCH]})  # both known
    with pytest.raises(TriggerError, match="unknown tool 'gdrive_nope'"):
        engine.register({"triggers": [{"id": "bad", "when": {"type": "action", "tool": "gsheets_values_update"},
                                       "actions": [{"type": "tool", "tool": "gdrive_nope"}]}]})
    with pytest.raises(TriggerError, match="when.tool names unknown tool"):
        engine.register({"triggers": [{"id": "bad2", "when": {"type": "action", "tool": "nope"}, "actions": []}]})
    # before discovery (no server tools) the check is skipped
    lazy = _make_engine(monkeypatch, {})
    lazy.register({"triggers": [{"id": "ok", "when": {"type": "action", "tool": "anything"},
                                 "actions": [{"type": "tool", "tool": "whatever"}]}]})


# ---------------------------------------------------------------- firing with ctx (E1, E5, E7, E9)

@pytest.mark.asyncio
async def test_hero_touch_repeats_with_templated_args(monkeypatch):
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"id": a["fileId"], "modifiedTime": "t"}})
    engine.register({"triggers": [TOUCH]})
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1", "range": "A1", "values": [[1]]},
                        _result(json.dumps({"spreadsheetId": "S1", "updatedCells": 1})))
    await _settle(engine)
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S2", "range": "A1", "values": [[1]]},
                        _result(json.dumps({"spreadsheetId": "S2", "updatedCells": 1})))
    await _settle(engine)
    gw = engine._test_gw
    assert gw.calls == [("gdrive_touch_file", {"fileId": "S1"}), ("gdrive_touch_file", {"fileId": "S2"})]
    trig = engine._triggers["touch"]
    assert trig["status"] == "armed" and trig["fire_count"] == 2
    kinds = [e["kind"] for e in engine._events]
    assert kinds.count("fired") == 2 and kinds.count("action_ok") == 2
    ok = [e for e in engine._events if e["kind"] == "action_ok"][0]
    assert ok["detail"] == {"tool": "gdrive_touch_file", "args": {"fileId": "S1"}}
    assert ok["changelog_id_after"] > ok["changelog_id_before"]
    # E7: the write is in the trajectory, not only the trigger_fired marker
    internal = [e for e in gw.events if e["event_type"] == "internal_tool_call"]
    assert [e["arguments"] for e in internal] == [{"fileId": "S1"}, {"fileId": "S2"}]
    assert all(e["ok"] and e["trigger_id"] == "touch" and e["changelog_id_after"] > e["changelog_id_before"] for e in internal)
    assert [e["event_type"] for e in gw.events].count("trigger_fired") == 2


@pytest.mark.asyncio
async def test_back_to_back_calls_while_firing_are_queued_not_dropped(monkeypatch):
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"id": a["fileId"]}},
                          delays={"gdrive_touch_file": 0.15})
    engine.register({"triggers": [TOUCH]})
    for sid in ("S1", "S2", "S3"):
        engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": sid}, _result("{}"))
    assert engine._triggers["touch"]["status"] == "queued"
    assert engine.state()["triggers"][0]["pending"] == 2
    queued = [e for e in engine._events if e["kind"] == "detected" and e.get("queued")]
    assert [e["queued"] for e in queued] == [1, 2]
    await _settle(engine, timeout=3.0)
    assert [a["fileId"] for _, a in engine._test_gw.calls] == ["S1", "S2", "S3"]
    assert engine._triggers["touch"]["fire_count"] == 3 and engine._triggers["touch"]["status"] == "armed"


@pytest.mark.asyncio
async def test_one_shot_trigger_still_fires_once_and_skips_while_firing(monkeypatch):
    once = dict(TOUCH, id="once", when={"type": "action", "tool": "gsheets_values_update"})
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"id": a["fileId"]}},
                          delays={"gdrive_touch_file": 0.1})
    engine.register({"triggers": [once]})
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"}, _result("{}"))
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S2"}, _result("{}"))  # skipped: firing, no repeat
    await _settle(engine)
    assert engine._triggers["once"]["status"] == "fired" and engine._triggers["once"]["fire_count"] == 1
    assert engine._test_gw.calls == [("gdrive_touch_file", {"fileId": "S1"})]
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S3"}, _result("{}"))
    await asyncio.sleep(0.05)
    assert len(engine._test_gw.calls) == 1


@pytest.mark.asyncio
async def test_a_refused_call_fires_nothing_and_a_refused_mirror_is_action_failed(monkeypatch):
    """Refusals reach the engine as `isError`, including the ones a server handles and returns as
    JSON — BaseService's flag_handled_errors sets the flag at the wire boundary."""
    engine = _make_engine(monkeypatch, {
        "gdrive_touch_file": lambda a: _result(json.dumps({"error": {"code": 404, "message": "File not found"}}),
                                               is_error=True)})
    engine.register({"triggers": [TOUCH]})
    # a refused provoking call mirrors nothing
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"},
                        _result(json.dumps({"error": {"code": 403, "message": "no permission"}}), is_error=True))
    await asyncio.sleep(0.05)
    assert engine._test_gw.calls == [] and engine._triggers["touch"]["status"] == "armed"
    # a mirror that is refused is action_failed, and the trigger re-arms
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"}, _result("{}"))
    await _settle(engine)
    kinds = [e["kind"] for e in engine._events]
    assert "action_failed" in kinds and "failed" in kinds and "fired" not in kinds
    failed = [e for e in engine._events if e["kind"] == "action_failed"][0]
    assert "File not found" in failed["detail"] and failed["tool"] == "gdrive_touch_file"
    assert engine._triggers["touch"]["status"] == "armed"
    assert engine._triggers["touch"]["failure_count"] == 1 and engine._triggers["touch"]["last_failure_at"]
    internal = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"]
    assert len(internal) == 1 and internal[0]["ok"] is False
    # the next call fires again (no permanent park)
    engine._test_gw.responses["gdrive_touch_file"] = lambda a: {"id": a["fileId"]}
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"}, _result("{}"))
    await _settle(engine)
    assert engine._triggers["touch"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_a_one_shot_failure_rearms_and_the_tally_survives_the_later_success(monkeypatch):
    """No trigger parks on failure any more, so the sticky per-trigger tally is the only thing that
    still answers "did this coupling ever break?" after a later fire succeeds."""
    once = dict(TOUCH, id="once", when={"type": "action", "tool": "gsheets_values_update"})
    engine = _make_engine(monkeypatch, {
        "gdrive_touch_file": lambda a: _result(json.dumps({"error": {"code": 404, "message": "nope"}}),
                                               is_error=True)})
    engine.register({"triggers": [once]})
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"}, _result("{}"))
    await _settle(engine)
    trig = engine._triggers["once"]
    assert trig["status"] == "armed" and trig["fire_count"] == 0        # retryable, not parked
    assert trig["failure_count"] == 1 and trig["last_failure_at"]
    failed = [e for e in engine._events if e["kind"] == "failed"][0]
    assert "rearmed" not in failed and "dropped" not in failed         # the policy keys are gone
    # a one-shot that finally succeeds retires, and the failure is still on the record
    engine._test_gw.responses["gdrive_touch_file"] = lambda a: {"id": a["fileId"]}
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S2"}, _result("{}"))
    await _settle(engine)
    row = {t["id"]: t for t in engine.state()["triggers"]}["once"]
    assert row["status"] == "fired" and row["fire_count"] == 1 and row["failure_count"] == 1


@pytest.mark.asyncio
async def test_unresolved_placeholder_is_action_failed_not_a_literal_send(monkeypatch):
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"id": a.get("fileId")}})
    engine.register({"triggers": [TOUCH]})
    engine.on_tool_call("default", "gsheets_values_update", {"range": "A1"}, _result("{}"))  # no spreadsheetId
    await _settle(engine)
    assert engine._test_gw.calls == []
    failed = [e for e in engine._events if e["kind"] == "action_failed"]
    assert failed and "unresolved placeholder ${args.spreadsheetId}" in failed[0]["detail"]
    assert engine._triggers["touch"]["status"] == "armed"  # rearm


@pytest.mark.asyncio
async def test_create_mirror_uses_result_where_barrier_and_verify(monkeypatch):
    registered: dict = {}

    def register(a):
        registered[a["id"]] = a
        return {"registered": True, "file": {"id": a["id"], "name": a["name"]}}

    engine = _make_engine(monkeypatch, {
        "gdrive_register_native_file": register,
        "gdrive_get_file_metadata": lambda a: ({"id": a["fileId"], "name": registered[a["fileId"]]["name"]}
                                               if a["fileId"] in registered else {"error": {"code": 404, "message": "not found"}}),
    })
    engine.register({"triggers": [CREATE]})
    ack = {"documentId": "doc-9", "title": "Plan", "owners": ["julia@x.co"], "createdTime": "2026-07-06T09:00:00Z",
           "driveFileId": None}
    pending, timeout_s = engine.on_tool_call("default", "gdocs_create_document", {"title": "Plan"},
                                             _result(json.dumps(ack)))
    # the barrier hands the caller the fire to hold this call for; no explicit ask -> gateway default
    assert len(pending) == 1 and isinstance(pending[0], asyncio.Task)
    assert timeout_s == engine._test_gw.TRIGGER_BARRIER_TIMEOUT_S
    await asyncio.wait(pending, timeout=2)
    assert registered == {"doc-9": {"id": "doc-9", "name": "Plan", "mimeType": "application/vnd.google-apps.document",
                                    "ownerEmail": "julia@x.co", "createdTime": "2026-07-06T09:00:00Z"}}
    kinds = [e["kind"] for e in engine._events]
    assert "verify_ok" in kinds and "fired" in kinds
    assert engine._test_gw.calls[-1] == ("gdrive_get_file_metadata", {"fileId": "doc-9"})
    # a two-step create (agent passed driveFileId) is not mirrored: `where driveFileId exists:false`
    engine.on_tool_call("default", "gdocs_create_document", {"title": "Two", "driveFileId": "f_1"},
                        _result(json.dumps(dict(ack, documentId="doc-2", driveFileId="f_1"))))
    await asyncio.sleep(0.05)
    assert "doc-2" not in registered
    # a trigger with no barrier hands back nothing to wait on
    engine.register({"triggers": [TOUCH]})
    assert engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S"}, _result("{}")) == ([], None)
    await _settle(engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("first,second,expected", [
    (None, None, 30.0),   # both take the default
    (None, 5, 30.0),      # a shorter explicit ask must not cut the default-bound one short
    (None, 120, 120.0),   # a longer one widens past the default
    (5, 9, 9),            # two explicit asks: the widest wins
])
async def test_mixed_barriers_hold_the_call_for_the_widest_bound(monkeypatch, first, second, expected):
    """One call is held once, so its bound is the widest of its barriers'; an omitted
    `timeout_seconds` is a request for the gateway default, not agreement with a neighbour."""
    def _barrier(timeout):
        return {"at": "provoking_call"} if timeout is None else {"at": "provoking_call",
                                                                 "timeout_seconds": timeout}
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"ok": True}})
    default_bound = {"id": "b_first", "when": {"type": "action", "tool": "gsheets_values_update", "repeat": True},
                     "barrier": _barrier(first),
                     "actions": [{"type": "tool", "tool": "gdrive_touch_file", "args": {"fileId": "d"}}]}
    explicit_bound = dict(default_bound, id="b_second", barrier=_barrier(second))
    engine.register({"triggers": [default_bound, explicit_bound]})

    pending, timeout_s = engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"},
                                             _result("{}"))
    assert len(pending) == 2
    # the widest of {gateway default, explicit ask} wins, whichever order they were registered in
    assert timeout_s == expected
    await _settle(engine)


@pytest.mark.asyncio
async def test_where_over_result_paths(monkeypatch):
    spec = {"id": "invite", "when": {"type": "action", "tool": "gcal_create_event", "repeat": True,
                                     "where": {"result.guestEmails": {"exists": True}, "sendUpdates": {"equals": "all"}}},
            "actions": [{"type": "tool", "tool": "gmail_deliver_message", "args": {"to": "${result.guestEmails}"}}]}
    engine = _make_engine(monkeypatch, {"gmail_deliver_message": lambda a: {"delivered": len(a["to"])}})
    engine.register({"triggers": [spec]})
    engine.on_tool_call("default", "gcal_create_event", {"sendUpdates": "all"}, _result(json.dumps({"guestEmails": []})))
    engine.on_tool_call("default", "gcal_create_event", {"sendUpdates": "none"}, _result(json.dumps({"guestEmails": ["g"]})))
    await asyncio.sleep(0.05)
    assert engine._test_gw.calls == []
    engine.on_tool_call("default", "gcal_create_event", {"sendUpdates": "all"}, _result(json.dumps({"guestEmails": ["g1", "g2"]})))
    await _settle(engine)
    assert engine._test_gw.calls == [("gmail_deliver_message", {"to": ["g1", "g2"]})]


@pytest.mark.asyncio
async def test_non_json_result_and_state_triggers_keep_working(monkeypatch):
    # a plain-text result is still a provoking call; ${result} would be the text itself
    spec = {"id": "txt", "when": {"type": "action", "tool": "t", "repeat": True},
            "actions": [{"type": "tool", "tool": "log", "args": {"msg": "got ${result}"}}]}
    engine = _make_engine(monkeypatch, {"log": lambda a: {"ok": True}})
    engine.register({"triggers": [spec]})
    engine.on_tool_call("default", "t", {}, _result("hello"))
    await _settle(engine)
    assert engine._test_gw.calls == [("log", {"msg": "got hello"})]


def test_known_tools_requires_completed_discovery(monkeypatch):
    engine = _make_engine(monkeypatch, {}, known_tools=["a"])
    assert engine._known_tools() == {"a"}
    engine._test_gw._tools_discovered = False
    assert engine._known_tools() is None


@pytest.mark.asyncio
async def test_a_failing_trigger_drains_its_queue_and_a_refused_call_evaluates_nothing(monkeypatch):
    disarm = dict(TOUCH, id="disarm")
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: _result("{}", is_error=True),
                                        "probe": lambda a: {"v": "hit"}},
                          delays={"gdrive_touch_file": 0.1})
    engine.register({"triggers": [disarm,
                                  {"id": "sensor", "when": {"type": "state", "check": {"tool": "probe", "extract": "v",
                                                                                          "predicate": {"equals": "hit"}}},
                                   "actions": []}]})
    for sid in ("S1", "S2", "S3"):
        engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": sid}, _result("{}"))
    await _settle(engine)
    failed = [e for e in engine._events if e["kind"] == "failed" and e["trigger_id"] == "disarm"]
    # every queued detection is served rather than thrown away with the trigger: 3 calls -> 3 failures
    assert len(failed) == 3 and all("dropped" not in e for e in failed)
    assert engine._triggers["disarm"]["status"] == "armed" and not engine._triggers["disarm"]["pending"]
    assert engine._triggers["disarm"]["failure_count"] == 3
    assert engine._triggers["sensor"]["status"] == "fired"  # state sensors still evaluated (the provoking calls succeeded)
    # a refused provoking call is not a write the engine reacts to at all: no action detection,
    # and not even a state re-evaluation
    engine._triggers["sensor"]["status"] = "armed"
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S9"},
                        _result(json.dumps({"error": {"code": 403, "message": "no"}}), is_error=True))
    await _settle(engine)
    assert engine._triggers["sensor"]["status"] == "armed"
    assert len([e for e in engine._events if e["kind"] == "detected" and e["trigger_id"] == "disarm"]) == 3


@pytest.mark.asyncio
async def test_fire_settles_status_if_dependents_resolution_raises(monkeypatch):
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"id": a["fileId"]}})
    engine.register({"triggers": [TOUCH]})

    def boom(tid):
        raise RuntimeError("anchor failure")
    monkeypatch.setattr(engine, "_resolve_dependents", boom)
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"}, _result("{}"))
    await _settle(engine)
    trig = engine._triggers["touch"]
    assert trig["status"] == "armed" and trig["task"] is None and trig["fire_count"] == 1
    # the unwedge is not silent: it says the fire never settled and what it discarded
    unsettled = [e for e in engine._events if e["kind"] == "failed" and e.get("detail") == "firing did not settle"]
    assert len(unsettled) == 1 and unsettled[0]["dropped"] == 0 and trig["failure_count"] == 1


def test_optional_placeholders_drop_missing_keys_instead_of_failing():
    ctx = {"args": {"type": "user", "emailAddress": "r@x.co", "role": "reader"}, "result": {"file": {"name": "Plan"}}}
    out = _template({"kind": "${args.type}", "emailAddress": "${args.emailAddress?}", "domain": "${args.domain?}",
                     "role": "${args.role?}", "note": "for ${args.domain?}:${args.emailAddress?}",
                     "list": ["${args.emailAddress?}", "${args.domain?}"]}, {}, ctx)
    assert out == {"kind": "user", "emailAddress": "r@x.co", "role": "reader", "note": "for :r@x.co", "list": ["r@x.co"]}
    with pytest.raises(TemplateError):  # a required placeholder next to an optional one still fails loud
        _template({"a": "${args.domain?}", "b": "${args.missing}"}, {}, ctx)


def test_optional_placeholder_validates_like_a_required_one(monkeypatch):
    engine = _make_engine(monkeypatch, {})
    engine.register({"triggers": [{"id": "ok", "when": {"type": "action", "tool": "t"},
                                   "actions": [{"type": "tool", "tool": "t2", "args": {"a": "${args.x?}", "b": "${result.y[0]?}"}}]}]})
    with pytest.raises(TriggerError, match="only action triggers carry"):
        engine.register({"triggers": [{"id": "bad", "when": {"type": "time", "at": "PT1H"},
                                       "actions": [{"type": "tool", "tool": "t2", "args": {"a": "${args.x?}"}}]}]})


@pytest.mark.asyncio
async def test_nl_action_verify_can_template_the_provoking_call(monkeypatch):
    """An `nl` action's verify is templated against the same provoking-call context a `tool` action's is.
    Without the context threaded through _run_nl, `${result.*}` here raises TemplateError and the trigger
    fails outright instead of verifying (and retrying)."""
    seen: list[dict] = []
    engine = _make_engine(monkeypatch, {
        "gdrive_get_file_metadata": lambda a: (seen.append(a) or {"id": a["fileId"]}),
    })
    spec = {"id": "nl-verify",
            "when": {"type": "action", "tool": "gdocs_create_document", "repeat": True},
            "actions": [{"type": "nl", "instruction": "mirror the doc into Drive",
                         "verify": {"tool": "gdrive_get_file_metadata",
                                    "args": {"fileId": "${result.documentId}"},
                                    "extract": "id", "predicate": {"exists": True}}}]}
    engine.register({"triggers": [spec],
                     "executor": {"a2a_url": "http://executor.local", "role": "executor"}})

    async def fake_executor(self, instruction, context_id):
        return "done"
    monkeypatch.setattr(TriggerEngine, "_call_executor", fake_executor)

    engine.on_tool_call("default", "gdocs_create_document", {"title": "Plan"},
                        _result(json.dumps({"documentId": "doc-7"})))
    await _settle(engine)
    assert seen == [{"fileId": "doc-7"}], seen
    kinds = [e["kind"] for e in engine._events]
    assert "verify_ok" in kinds and "fired" in kinds, kinds


@pytest.mark.asyncio
async def test_a_refused_verification_does_not_read_as_satisfied(monkeypatch):
    """A refused verification read is not data to predicate over — its payload is a dict, so an
    `exists: true` over it would otherwise match and emit verify_ok."""
    engine = _make_engine(monkeypatch, {
        "gdrive_register_native_file": lambda a: {"registered": True},
        "gdrive_get_file_metadata": lambda a: _result(json.dumps({"ok": False, "error": "permission denied"}),
                                                      is_error=True),
    })
    spec = {"id": "verify-refused",
            "when": {"type": "action", "tool": "gdocs_create_document", "repeat": True},
            "actions": [{"type": "tool", "tool": "gdrive_register_native_file", "args": {"id": "x"},
                         "verify": {"tool": "gdrive_get_file_metadata", "args": {"fileId": "x"},
                                    "predicate": {"exists": True}}}]}
    engine.register({"triggers": [spec]})
    engine.on_tool_call("default", "gdocs_create_document", {"title": "Plan"}, _result("{}"))
    await _settle(engine)
    kinds = [e["kind"] for e in engine._events]
    assert "verify_failed" in kinds and "verify_ok" not in kinds, kinds
    assert "fired" not in kinds, kinds


@pytest.mark.asyncio
async def test_a_refused_state_sensor_does_not_fire(monkeypatch):
    """Same rule on the detection side: a state trigger whose sensor is refused must stay armed."""
    engine = _make_engine(monkeypatch, {
        "gdrive_list_files": lambda a: _result(json.dumps({"ok": False, "error": "backend down"}), is_error=True),
        "gmail_deliver_message": lambda a: {"delivered": 1},
    })
    spec = {"id": "state-refused",
            "when": {"type": "state", "interval_seconds": 0,
                     "check": {"steps": [{"tool": "gdrive_list_files", "args": {},
                                          "predicate": {"exists": True}}]}},
            "actions": [{"type": "tool", "tool": "gmail_deliver_message", "args": {"to": "x@y.co"}}]}
    engine.register({"triggers": [spec]})
    assert await engine._run_check(spec["when"]["check"]) is False
    assert "gmail_deliver_message" not in [c[0] for c in engine._test_gw.calls]


# --- Trajectory echo is an allowlist, not a size bound -------------------------------------
# The engine mirrors writes on the agent's behalf, so a mirror's args are request bodies:
# `contentB64` is a whole document, `text` a Slack message. Those must not reach the trajectory,
# and neither may the tool's response body. What a grader does need — which object was written —
# survives, because identifier/metadata keys stay in the clear.

IMPORT_DOC = {"id": "import", "when": {"type": "action", "tool": "gdrive_create_file", "repeat": True},
              "actions": [{"type": "tool", "tool": "gdocs_import_document",
                           "args": {"documentId": "${result.id}", "title": "${args.name}",
                                    "mimeType": "application/vnd.google-apps.document",
                                    "contentB64": "${args.content}",
                                    "ownerEmail": "${result.owner}"}}]}


def _fire_import(monkeypatch, body: str, owner: str = "ada@example.com"):
    engine = _make_engine(monkeypatch, {"gdocs_import_document": lambda a: {"ok": True}})
    engine.register({"triggers": [IMPORT_DOC]})
    engine.on_tool_call("default", "gdrive_create_file", {"name": "Q3 plan", "content": body},
                        _result(json.dumps({"id": "DOC1", "owner": owner})))
    return engine


@pytest.mark.asyncio
async def test_a_mirrors_payload_never_reaches_the_trajectory(monkeypatch):
    """The document body is redacted to a shape; the id and title it was written under are not."""
    secret = "CONFIDENTIAL board minutes " * 40
    engine = _fire_import(monkeypatch, secret)
    await _settle(engine)

    event = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"][-1]
    args = event["arguments"]

    assert args["contentB64"] == {"redacted": {"type": "str", "len": len(secret)}}
    # Structure survives redaction: the arg names a mirror sent stay reviewable.
    assert set(args) == {"documentId", "title", "mimeType", "contentB64", "ownerEmail"}
    assert args["documentId"] == "DOC1" and args["title"] == "Q3 plan"
    assert args["mimeType"] == "application/vnd.google-apps.document"
    # Not merely truncated — no fragment of the body anywhere in the serialized event.
    assert "CONFIDENTIAL" not in json.dumps(event)


@pytest.mark.asyncio
async def test_a_short_payload_is_redacted_too(monkeypatch):
    """The old bound echoed anything under 2000 chars verbatim, which is where the leak lived."""
    engine = _fire_import(monkeypatch, "layoffs on the 14th")
    await _settle(engine)

    event = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"][-1]
    assert event["arguments"]["contentB64"] == {"redacted": {"type": "str", "len": 19}}
    assert "layoffs" not in json.dumps(event)


@pytest.mark.asyncio
async def test_recipients_are_not_echoed_even_though_they_are_scalars(monkeypatch):
    """An address is PII, so it is redacted despite being a short scalar a grader might like."""
    engine = _fire_import(monkeypatch, "body", owner="ada@example.com")
    await _settle(engine)

    event = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"][-1]
    assert event["arguments"]["ownerEmail"] == {"redacted": {"type": "str", "len": 15}}
    assert "ada@example.com" not in json.dumps(event)


@pytest.mark.asyncio
async def test_the_response_body_is_reduced_to_a_length_and_a_digest(monkeypatch):
    """`result_text` carried up to 2000 chars of the tool's response; nothing carries it now."""
    engine = _make_engine(monkeypatch, {"gdrive_touch_file": lambda a: {"secretField": "do-not-log"}})
    engine.register({"triggers": [TOUCH]})
    engine.on_tool_call("default", "gsheets_values_update", {"spreadsheetId": "S1"}, _result("{}"))
    await _settle(engine)

    event = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"][-1]

    assert "result_text" not in event
    assert event["result_len"] == len(json.dumps({"secretField": "do-not-log"}))
    assert len(event["result_sha256"]) == 16
    assert "do-not-log" not in json.dumps(event)
    # Still enough to tell one mirror's response from another's.
    assert event["ok"] is True and event["arguments"] == {"fileId": "S1"}


@pytest.mark.asyncio
async def test_a_nested_payload_is_redacted_at_depth(monkeypatch):
    """Keys are matched at every depth, so nesting is not a way around the allowlist."""
    engine = _make_engine(monkeypatch, {"svc_write": lambda a: {"ok": True}})
    engine.register({"triggers": [{
        "id": "nested", "when": {"type": "action", "tool": "svc_touch", "repeat": True},
        "actions": [{"type": "tool", "tool": "svc_write",
                     "args": {"id": "${args.id}",
                              "file": {"name": "${args.name}", "content": "${args.body}"},
                              "recipients": ["${args.to}"]}}]}]})
    engine.on_tool_call("default", "svc_touch",
                        {"id": "X1", "name": "notes", "body": "inner secret", "to": "eve@example.com"},
                        _result("{}"))
    await _settle(engine)

    args = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"][-1]["arguments"]

    assert args["id"] == "X1"
    assert args["file"]["name"] == "notes"
    assert args["file"]["content"] == {"redacted": {"type": "str", "len": 12}}
    # A list element inherits its parent key, so a non-allowlisted list is redacted per element.
    assert args["recipients"] == [{"redacted": {"type": "str", "len": 15}}]


@pytest.mark.asyncio
async def test_an_allowlisted_key_holding_a_payload_is_still_redacted(monkeypatch):
    """`_ECHO_VALUE_MAX` stops a payload smuggled under an identifier-shaped name."""
    engine = _make_engine(monkeypatch, {"svc_write": lambda a: {"ok": True}})
    engine.register({"triggers": [{
        "id": "fat-name", "when": {"type": "action", "tool": "svc_touch", "repeat": True},
        "actions": [{"type": "tool", "tool": "svc_write", "args": {"name": "${args.name}"}}]}]})
    engine.on_tool_call("default", "svc_touch", {"name": "x" * 500}, _result("{}"))
    await _settle(engine)

    args = [e for e in engine._test_gw.events if e["event_type"] == "internal_tool_call"][-1]["arguments"]
    assert args["name"] == {"redacted": {"type": "str", "len": 500}}


@pytest.mark.asyncio
async def test_the_failure_record_redacts_args_but_keeps_the_reason(monkeypatch):
    """A broken mirror still has to be diagnosable: the refusal text is a bounded diagnostic."""
    engine = _make_engine(
        monkeypatch,
        {"gdocs_import_document": lambda a: _result("Import failed: quota exceeded", is_error=True)})
    engine.register({"triggers": [IMPORT_DOC]})
    engine.on_tool_call("default", "gdrive_create_file", {"name": "Q3 plan", "content": "secret body"},
                        _result(json.dumps({"id": "DOC1", "owner": "ada@example.com"})))
    await _settle(engine)

    failed = [e for e in engine._events if e["kind"] == "action_failed"][-1]

    assert "quota exceeded" in failed["detail"]
    assert failed["args"]["contentB64"] == {"redacted": {"type": "str", "len": 11}}
    assert failed["args"]["documentId"] == "DOC1"
    assert "secret body" not in json.dumps(failed)


# --- placeholder shapes the engine must not silently pass through -------------------------------

_SHAPED_RE = re.compile(r"\$\{[^{}]*\}?")  # anything placeholder-shaped, written from the spec


def _resolvable(token: str) -> bool:
    """The documented placeholder contract, restated here rather than imported on purpose.

    Importing the engine's own pattern would make this oracle agree with whatever the engine does,
    including being wrong -- which is exactly how the trailing-space form survived. `${<path>}`,
    where the path is word characters with `.` separators and `[n]` / `[*]` indices and an optional
    trailing `?`; a dotted path must be rooted at args/result, and a single bare word is a check
    binding name."""
    m = re.fullmatch(r"\$\{(\w+(?:\.\w+|\[\d+\]|\[\*\])*)\??\}", token)
    if m is None:
        return False
    name = m.group(1)
    return (name.split(".")[0].split("[")[0] in ("args", "result")
            or re.fullmatch(r"\w+", name) is not None)


def _placeholder_mutations() -> list[str]:
    """Every single-character corruption of one valid placeholder, plus shapes a human types.

    Generated rather than hand-listed: the value is covering the typo nobody thought of."""
    valid = "${result.email_id}"
    body = valid[2:-1]
    out = {valid}
    for i in range(len(valid) + 1):
        out.add(valid[:i] + " " + valid[i:])                 # a space slipped in anywhere
    for i in range(len(body)):
        out.add("${" + body[:i] + "-" + body[i + 1:] + "}")  # a character mistyped
    out.update({"${}", "${" + body, "$ {" + body + "}", valid + "}",
                "${" + body.upper() + "}", "${" + body + "?}"})
    return sorted(out)


@pytest.mark.parametrize("candidate", _placeholder_mutations())
def test_registration_rejects_every_placeholder_shape_it_cannot_resolve(monkeypatch, candidate):
    """Accepted-but-never-substituted must not be a reachable outcome.

    The strict pattern is both the finder and the resolver, so a shape it cannot parse used to be
    invisible twice: never rejected here, never substituted at fire time, delivered to the target
    tool as literal `${...}` text. On a server that does not flag handled errors that is a silent
    wrong write, and the run still scores."""
    engine = _make_engine(monkeypatch, {"move_email": lambda a: "{}"})
    body = {"watch_roles": ["default"],
            "triggers": [{"id": "m", "when": {"type": "action", "tool": "send_email"},
                          "actions": [{"type": "tool", "tool": "move_email",
                                       "args": {"email_id": candidate}}]}]}
    if all(_resolvable(tok) for tok in _SHAPED_RE.findall(candidate)):
        engine.register(body)  # nothing placeholder-shaped, or every shape resolves
        assert engine._triggers["m"]["spec"]["actions"][0]["args"]["email_id"] == candidate
        return
    with pytest.raises(TriggerError):
        engine.register(body)
    assert engine._triggers == {}, "a rejected spec must register nothing"


def test_an_inert_placeholder_is_rejected_wherever_it_hides(monkeypatch):
    """Action args (nested in a list), a verify step, and a state check all share one scan."""
    bad = "${result.email_id }"
    engine = _make_engine(monkeypatch, {"move_email": lambda a: "{}", "probe": lambda a: "{}"})
    nested = {"id": "a", "when": {"type": "action", "tool": "send_email"},
              "actions": [{"type": "tool", "tool": "move_email",
                           "args": {"batch": [{"email_id": bad}]}}]}
    verified = {"id": "b", "when": {"type": "action", "tool": "send_email"},
                "actions": [{"type": "tool", "tool": "move_email", "args": {},
                             "verify": {"tool": "probe", "args": {"q": bad},
                                        "predicate": {"exists": True}}}]}
    # A state trigger cannot carry `result.*` at all, so this also pins the order: the shape is
    # reported as unparseable rather than as out-of-scope, which is the more actionable error.
    stated = {"id": "c", "when": {"type": "state", "check": {"tool": "probe", "args": {"q": bad},
                                                             "predicate": {"exists": True}}},
              "actions": [{"type": "tool", "tool": "move_email", "args": {}}]}
    for spec in (nested, verified, stated):
        with pytest.raises(TriggerError, match="not a placeholder"):
            engine.register({"watch_roles": ["default"], "triggers": [spec]})
    assert engine._triggers == {}


@pytest.mark.parametrize("instruction, rejected", [
    ("move ${result.email_id} to DRAFT", True),          # meant to template, never will
    ("move ${result.email_id } to DRAFT", True),         # malformed, still clearly meant to
    ("read ${args.subject} first", True),
    ("move ${result .email_id} to DRAFT", True),         # space before the dot: still the attempt
    ("move ${result. email_id} to DRAFT", True),         # and after it
    ("move ${ result.email_id } to DRAFT", True),
    ("write the value of ${HOME} into the doc", False),  # prose: not a placeholder shape at all
    ("mention ${SOME_TEMPLATE} verbatim", False),
    ("move the last email to DRAFT", False),
])
def test_an_nl_instruction_rejects_only_placeholders_it_would_never_substitute(monkeypatch, instruction, rejected):
    """nl instructions are prose handed to a model, and prose legitimately contains brace syntax.

    Only a context-rooted shape is unambiguous: `${args.*}` / `${result.*}` can only mean an author
    expecting templating, which nl never does. Rejecting every `${...}` would 400 an instruction that
    merely mentions `${HOME}`."""
    engine = _make_engine(monkeypatch, {})
    spec = {"id": "n", "when": {"type": "action", "tool": "send_email"},
            "actions": [{"type": "nl", "instruction": instruction}]}
    body = {"watch_roles": ["default"], "triggers": [spec],
            "executor": {"a2a_url": "http://executor.local", "role": "executor"}}
    if not rejected:
        engine.register(body)
        assert engine._triggers["n"]["spec"]["actions"][0]["instruction"] == instruction
        return
    with pytest.raises(TriggerError, match="not templated"):
        engine.register(body)
    assert engine._triggers == {}
