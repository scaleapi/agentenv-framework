"""Unit tests for the gateway trigger engine (urn:agentenv:triggers/v1)."""
from __future__ import annotations

import asyncio
import json

import pytest
from mcp.types import CallToolResult, TextContent

from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError


class _FakeGateway:
    def __init__(self, responses: dict) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []
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
        return "event_test"

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

    async def fake_internal_call(self, tool_name, arguments):
        gw.calls.append((tool_name, dict(arguments)))
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
    assert st["config"] == {"watch_roles": [], "executor_configured": False}
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
async def test_sensor_trigger_and_state_shape(engine):
    engine.register(_registration())
    engine._test_state["doc_text"] = "rate is $26/hr now"
    engine.on_tool_call("default", "gdocs_batch_update", {"documentId": "doc-1"}, _result())
    await asyncio.sleep(0.3)
    assert engine._triggers["sensor"]["status"] == "fired"
    state = engine.state()
    assert "sensor" in {t["id"] for t in state["triggers"]}
    assert state["config"] == {"watch_roles": ["default"], "executor_configured": False}
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

    class _FakeSession:
        async def call_tool(self, name, arguments):
            return _result()

    async def _fake_get_step_session(server_url):
        return _FakeSession()

    gw._get_step_session = _fake_get_step_session  # type: ignore[assignment]
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
