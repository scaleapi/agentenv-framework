"""Website triggers (`when.type: website`): a request the gateway proxies to a website provokes triggers like a tool call."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from agent_env.env.gateway import gateway as gateway_module
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS, TRIGGER_WEBSITE
from agent_env.env.gateway.gateway import GATEWAY_TRAJECTORY_FILE, Gateway
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError

_RealAsyncClient = httpx.AsyncClient  # captured before any test patches the module attribute
MESSAGE = {"to": "dana@example.com", "subject": "Lunch?"}


class _FakeGateway:
    def __init__(self, responses: dict) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []
        self.events: list[dict] = []
        self.rules: list[tuple[str, str, bool]] = []
        self.website_urls = {"mail": "http://mail-frontend", "shop": "http://shop-frontend"}
        self._role_rules_lock = asyncio.Lock()
        self._server_tools: dict = {}
        self._tool_server_urls: dict = {}
        self._clock = Clock()
        self.TRIGGER_BARRIER_TIMEOUT_S = 30.0

    def _apply_rule(self, role, tool, value):
        self.rules.append((role, tool, value))

    def _query_changelog_id(self):
        return None

    async def _log_event(self, event):
        self.events.append(dict(event))
        return f"event_{len(self.events)}"


def _result(payload) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))], isError=False)


def _make_engine(monkeypatch, responses: dict | None = None) -> TriggerEngine:
    gw = _FakeGateway(responses or {"mail_deliver_message": lambda a: {"delivered": True}})

    async def fake_internal_call(self, tool_name, arguments, role=None):
        gw.calls.append((tool_name, dict(arguments)))
        return _result(gw.responses[tool_name](arguments))

    monkeypatch.setattr(TriggerEngine, "_internal_call", fake_internal_call)
    engine = TriggerEngine(gw)
    engine._test_gw = gw
    return engine


async def _settle(engine, timeout=2.0):
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        if all(t["status"] not in ("firing", "queued") and not t["evaluating"] for t in engine._triggers.values()):
            return
    raise AssertionError(f"engine did not settle: {[(k, v['status']) for k, v in engine._triggers.items()]}")


def _detected(engine) -> list[tuple[str, dict]]:
    return [(e["trigger_id"], e["provoking"]) for e in engine._events if e["kind"] == "detected"]


REPLY = {"id": "reply", "when": {"type": "website", "app": "mail", "method": "post",
                                 "path": {"regex": "^/api/messages/?$"}, "repeat": True},
         "actions": [{"type": "tool", "tool": "mail_deliver_message",
                      "args": {"from": "${args.to}", "subject": "Re: ${args.subject}"}}]}


# ---------------------------------------------------------------- registration

@pytest.mark.parametrize("when,actions,fragment", [
    ({"type": "website"}, [], "when.app must be a non-empty string"),
    ({"type": "website", "app": "calendar"}, [], "unknown website 'calendar' (websites: ['mail', 'shop'])"),
    ({"type": "website", "app": "mail", "method": 5}, [], "when.method must be a non-empty HTTP method"),
    ({"type": "website", "app": "mail", "method": " "}, [], "when.method must be a non-empty HTTP method"),
    ({"type": "website", "app": "mail", "path": "/api"}, [], "when.path: predicate must have exactly one of"),
    ({"type": "website", "app": "mail", "status": {"regex": "("}}, [], "when.status: invalid regex"),
    ({"type": "website", "app": "mail", "repeat": "yes"}, [], "when.repeat must be a boolean"),
    ({"type": "website", "app": "mail", "where": {"result.id": {"exists": True}}}, [],
     "a website request carries args, not a result"),
    ({"type": "website", "app": "mail"}, [{"type": "tool", "tool": "t", "args": {"id": "${result.id}"}}],
     "a website request carries args, not a result"),
    ({"type": "website", "app": "mail"},
     [{"type": "tool", "tool": "t", "verify": {"tool": "v", "args": {"id": "${result.id?}"}, "predicate": {"exists": True}}}],
     "a website request carries args, not a result"),
    ({"type": "website", "app": "mail", "unless": {"tool": "t"}}, [], "when.unless is only valid on time triggers"),
    ({"type": "website", "app": "mail", "roles": []}, [], "when.roles must be a non-empty list"),
])
def test_registration_fails_loud(monkeypatch, when, actions, fragment):
    engine = _make_engine(monkeypatch)
    with pytest.raises(TriggerError) as e:
        engine.register({"triggers": [{"id": "x", "when": when, "actions": actions}]})
    assert fragment in str(e.value)


def test_registration_normalizes_the_method_and_state_echoes_the_request_matcher(monkeypatch):
    engine = _make_engine(monkeypatch)
    spec = {**REPLY, "when": {**REPLY["when"], "status": {"regex": "^2"}, "roles": ["default"],
                              "where": {"to": {"exists": True}}},
            "barrier": {"at": "provoking_call", "timeout_seconds": 5}}
    engine.register({"triggers": [spec]})
    engine.register({"triggers": [spec]})
    [state] = engine.state()["triggers"]
    assert state["type"] == "website" and state["barrier"] == {"at": "provoking_call", "timeout_seconds": 5}
    assert state["when"] == {"type": "website", "app": "mail", "method": "POST", "path": {"regex": "^/api/messages/?$"},
                             "status": {"regex": "^2"}, "roles": ["default"], "repeat": True}


def test_the_card_declares_website_triggers():
    triggers = next(e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_TRIGGERS_URI)
    assert TRIGGER_WEBSITE in triggers["params"]["features"]


# ---------------------------------------------------------------- detection

@pytest.mark.asyncio
async def test_a_matching_request_fires_with_its_args_and_is_attributed_in_the_log(monkeypatch):
    engine = _make_engine(monkeypatch)
    engine.register({"triggers": [REPLY]})
    assert engine.on_website_request("default", "mail", "POST", "/api/messages", 201, dict(MESSAGE)) == ([], None)
    await _settle(engine)

    assert engine._test_gw.calls == [("mail_deliver_message", {"from": "dana@example.com", "subject": "Re: Lunch?"})]
    request = {"app": "mail", "method": "POST", "path": "/api/messages", "status": 201}
    assert _detected(engine) == [("reply", {"source": "website", **request, "role": "default"})]
    [fired] = [e for e in engine._test_gw.events if e["event_type"] == "trigger_fired"]
    assert fired["provoking_request"] == request and fired["provoking_tool"] is None


@pytest.mark.asyncio
async def test_only_a_request_matching_app_method_path_status_and_where_provokes(monkeypatch):
    engine = _make_engine(monkeypatch)
    engine.register({"triggers": [{**REPLY, "when": {**REPLY["when"], "where": {"to": {"regex": r"@example\.com$"}}}}]})
    for app, method, path, status, args in [
        ("shop", "POST", "/api/messages", 201, MESSAGE),
        ("mail", "GET", "/api/messages", 200, MESSAGE),
        ("mail", "POST", "/api/messages/7/archive", 200, MESSAGE),
        ("mail", "POST", "/api/messages", 500, MESSAGE),
        ("mail", "POST", "/api/messages", 201, {**MESSAGE, "to": "lee@elsewhere.test"}),
        ("mail", "POST", "/api/messages/", 303, MESSAGE),
    ]:
        engine.on_website_request("default", app, method, path, status, dict(args))
        await _settle(engine)
    assert [p["path"] for _, p in _detected(engine)] == ["/api/messages/"]
    assert len(engine._test_gw.calls) == 1


@pytest.mark.asyncio
async def test_a_status_predicate_can_match_a_failed_request(monkeypatch):
    engine = _make_engine(monkeypatch)
    engine.register({"triggers": [{"id": "lockout", "when": {"type": "website", "app": "mail", "method": "POST",
                                                             "path": {"equals": "/login"}, "status": {"equals": 401},
                                                             "repeat": True},
                                   "actions": [{"type": "permission", "action": "disable", "role": "default",
                                                "tools": "*"}]}]})
    engine.on_website_request("default", "mail", "POST", "/login", 200, {})
    engine.on_website_request("default", "mail", "POST", "/login", 401, {})
    await _settle(engine)
    assert [p["status"] for _, p in _detected(engine)] == [401]
    assert engine._test_gw.rules == [("default", "*", True)]


@pytest.mark.asyncio
async def test_a_one_shot_fires_once_and_a_repeat_once_per_request(monkeypatch):
    engine = _make_engine(monkeypatch)
    once = {**REPLY, "id": "once", "when": {k: v for k, v in REPLY["when"].items() if k != "repeat"}}
    engine.register({"triggers": [REPLY, once]})
    for _ in range(3):
        engine.on_website_request("default", "mail", "POST", "/api/messages", 201, dict(MESSAGE))
        await _settle(engine)
    assert {tid: t["fire_count"] for tid, t in engine._triggers.items()} == {"reply": 3, "once": 1}


@pytest.mark.asyncio
async def test_requests_by_an_unwatched_role_provoke_only_the_triggers_that_list_it(monkeypatch):
    engine = _make_engine(monkeypatch)
    listed = {**REPLY, "id": "listed", "when": {**REPLY["when"], "roles": ["carol@example.com"]}}
    engine.register({"watch_roles": ["default"], "triggers": [REPLY, listed]})
    engine.on_website_request("carol@example.com", "mail", "POST", "/api/messages", 201, dict(MESSAGE))
    await _settle(engine)
    assert [(tid, p["role"]) for tid, p in _detected(engine)] == [("listed", "carol@example.com")]


@pytest.mark.asyncio
async def test_a_barrier_hands_back_the_fire_to_hold_the_response_for(monkeypatch):
    engine = _make_engine(monkeypatch)
    engine.register({"triggers": [{**REPLY, "barrier": {"at": "provoking_call", "timeout_seconds": 5}}]})
    pending, timeout_s = engine.on_website_request("default", "mail", "POST", "/api/messages", 201, dict(MESSAGE))
    assert len(pending) == 1 and timeout_s == 5
    await asyncio.wait(pending, timeout=2)
    assert engine._triggers["reply"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_only_an_unsafe_request_that_succeeded_prompts_the_state_triggers(monkeypatch):
    inbox: list[str] = []
    engine = _make_engine(monkeypatch, {"mail_list_messages": lambda a: {"subjects": list(inbox)}})
    engine.register({"triggers": [{"id": "sent", "when": {"type": "state", "check": {
        "tool": "mail_list_messages", "extract": "subjects", "predicate": {"regex": "Lunch"}}}, "actions": []}]})
    inbox.append("Lunch?")
    engine.on_website_request("default", "mail", "GET", "/api/messages", 200, {})
    engine.on_website_request("default", "mail", "POST", "/api/messages", 500, dict(MESSAGE))
    await _settle(engine)
    assert engine._test_gw.calls == [] and engine._triggers["sent"]["status"] == "armed"

    engine.on_website_request("default", "mail", "POST", "/api/messages", 201, dict(MESSAGE))
    await _settle(engine)
    assert engine._test_gw.calls == [("mail_list_messages", {})] and engine._triggers["sent"]["status"] == "fired"
    assert _detected(engine)[0][1]["source"] == "website"


# ---------------------------------------------------------------- through the gateway's /website proxy

class _Website:
    """A MockTransport upstream for the gateway's websites, answering every request with `status`."""

    def __init__(self, status: int = 201) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []

    def install(self, monkeypatch) -> None:
        async def body():
            yield b'{"ok": true}'

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(self.status, headers={"content-type": "application/json"}, content=body())

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))


def _gateway() -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                   website_urls={"mail": "http://mail-frontend"})


async def _request(gw: Gateway, method: str, target: str, *, headers: dict | None = None, body: bytes = b"",
                   order: list | None = None) -> int:
    """Run one request through the gateway's ASGI app; return its status, appending "response" to `order` at the status line."""
    url = urlsplit(target)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": unquote(url.path), "raw_path": url.path.encode(),
        "query_string": url.query.encode(), "root_path": "", "server": ("gateway", 80), "client": ("c", 1),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    done = asyncio.Event()
    received = False
    statuses: list[int] = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
            if order is not None:
                order.append("response")

    try:
        await gw._mcp.streamable_http_app()(scope, receive, send)
    finally:
        done.set()
    return statuses[0]


def _trajectory() -> list[dict]:
    with open(GATEWAY_TRAJECTORY_FILE) as f:
        return [json.loads(line) for line in f]


@pytest.mark.asyncio
async def test_a_barrier_holds_the_website_s_response_until_the_mirror_has_landed(monkeypatch):
    _Website(status=201).install(monkeypatch)
    order: list[str] = []
    calls: list[tuple[str, dict, str | None]] = []

    async def slow_internal_call(self, tool_name, arguments, role=None):
        await asyncio.sleep(0.05)
        calls.append((tool_name, dict(arguments), role))
        order.append("mirror")
        return _result({"delivered": True})

    monkeypatch.setattr(TriggerEngine, "_internal_call", slow_internal_call)
    gw = _gateway()
    gw._trigger_engine.register({"triggers": [{**REPLY, "barrier": {"at": "provoking_call"}}]})
    before = len(_trajectory())

    status = await _request(gw, "POST", "/website/mail/api/messages", headers={"content-type": "application/json"},
                            body=json.dumps(MESSAGE).encode(), order=order)

    assert status == 201 and order == ["mirror", "response"]
    assert calls == [("mail_deliver_message", {"from": "dana@example.com", "subject": "Re: Lunch?"}, None)]
    [fired] = [e for e in _trajectory()[before:] if e["event_type"] == "trigger_fired"]
    assert fired["provoking_request"] == {"app": "mail", "method": "POST", "path": "/api/messages", "status": 201}


@pytest.mark.asyncio
async def test_the_proxy_hands_the_engine_the_role_path_status_and_args_of_each_request(monkeypatch):
    website = _Website(status=200)
    website.install(monkeypatch)
    gw = _gateway()
    gw._trigger_engine.register({"triggers": [REPLY]})
    seen: list[tuple] = []
    gw._trigger_engine.on_website_request = lambda *a: seen.append(a) or ([], None)
    form = {"content-type": "application/x-www-form-urlencoded; charset=utf-8"}
    json_body = {"content-type": "application/json"}

    await _request(gw, "GET", "/website/mail")
    await _request(gw, "GET", "/website/mail/inbox?folder=work&tag=a&tag=b", headers={"AgentEnv-Role": "carol@example.com"})
    await _request(gw, "POST", "/website/mail/api/messages?draft=1", headers=json_body,
                   body=json.dumps({"to": "dana@example.com", "draft": "0"}).encode())
    await _request(gw, "POST", "/website/mail/compose", headers=form, body=b"subject=Hi+there&to=dana%40example.com")
    await _request(gw, "POST", "/website/mail/api/batch?op=send", headers=json_body, body=b"[1, 2]")
    await _request(gw, "PUT", "/website/mail/api/drafts/a%2Fb", headers=json_body, body=b"not json")
    assert await _request(gw, "GET", "/website/calendar/") == 404

    assert seen == [
        ("default", "mail", "GET", "/", 200, {}),
        ("carol@example.com", "mail", "GET", "/inbox", 200, {"folder": "work", "tag": ["a", "b"]}),
        ("default", "mail", "POST", "/api/messages", 200, {"draft": "0", "to": "dana@example.com"}),
        ("default", "mail", "POST", "/compose", 200, {"subject": "Hi there", "to": "dana@example.com"}),
        ("default", "mail", "POST", "/api/batch", 200, {"op": "send"}),
        ("default", "mail", "PUT", "/api/drafts/a/b", 200, {}),
    ]
    assert [r.content for r in website.requests][2:4] == [b'{"to": "dana@example.com", "draft": "0"}',
                                                          b"subject=Hi+there&to=dana%40example.com"]


@pytest.mark.asyncio
async def test_without_a_website_trigger_a_mutating_website_request_still_rechecks_the_state_triggers(monkeypatch):
    _Website(status=201).install(monkeypatch)
    calls: list[str] = []

    async def internal_call(self, tool_name, arguments, role=None):
        calls.append(tool_name)
        return _result({"subjects": ["Lunch?"]})

    monkeypatch.setattr(TriggerEngine, "_internal_call", internal_call)
    gw = _gateway()
    gw._trigger_engine.register({"triggers": [{"id": "sent", "actions": [], "when": {"type": "state", "check": {
        "tool": "mail_list_messages", "extract": "subjects", "predicate": {"regex": "Lunch"}}}}]})
    parsed: list[bytes] = []
    real_args = gateway_module._request_args
    monkeypatch.setattr(gateway_module, "_request_args", lambda request, body: parsed.append(body) or real_args(request, body))

    assert await _request(gw, "POST", "/website/mail/api/messages", headers={"content-type": "application/json"},
                          body=json.dumps(MESSAGE).encode()) == 201
    await asyncio.sleep(0.1)

    assert parsed == [] and calls == ["mail_list_messages"] and gw._trigger_engine._triggers["sent"]["status"] == "fired"


@pytest.mark.asyncio
async def test_with_neither_a_website_nor_a_state_trigger_the_proxy_never_reaches_the_engine(monkeypatch):
    _Website(status=201).install(monkeypatch)
    gw = _gateway()
    gw._trigger_engine.register({"triggers": [{"id": "later", "actions": [], "when": {"type": "time", "at": "PT1H"}}]})
    seen: list[tuple] = []
    real = gw._trigger_engine.on_website_request
    gw._trigger_engine.on_website_request = lambda *a: seen.append(a) or real(*a)

    assert await _request(gw, "POST", "/website/mail/api/messages", headers={"content-type": "application/json"},
                          body=json.dumps(MESSAGE).encode()) == 201

    assert seen == []


@pytest.mark.asyncio
async def test_with_a_website_trigger_a_mutating_website_request_rechecks_the_state_triggers(monkeypatch):
    _Website(status=201).install(monkeypatch)
    calls: list[str] = []

    async def internal_call(self, tool_name, arguments, role=None):
        calls.append(tool_name)
        return _result({"subjects": ["Lunch?"], "delivered": True})

    monkeypatch.setattr(TriggerEngine, "_internal_call", internal_call)
    gw = _gateway()
    gw._trigger_engine.register({"triggers": [
        {"id": "sent", "actions": [], "when": {"type": "state", "check": {
            "tool": "mail_list_messages", "extract": "subjects", "predicate": {"regex": "Lunch"}}}},
        {**REPLY, "when": {**REPLY["when"], "path": {"equals": "/never"}}}]})

    await _request(gw, "POST", "/website/mail/api/messages", headers={"content-type": "application/json"},
                   body=json.dumps(MESSAGE).encode())
    await asyncio.sleep(0.1)

    assert calls == ["mail_list_messages"] and gw._trigger_engine._triggers["sent"]["status"] == "fired"


@pytest.mark.asyncio
async def test_a_failing_trigger_evaluation_still_answers_the_request(monkeypatch):
    _Website(status=200).install(monkeypatch)
    gw = _gateway()
    gw._trigger_engine.register({"triggers": [REPLY]})

    def explode(*_):
        raise RuntimeError("boom")

    gw._trigger_engine.on_website_request = explode
    assert await _request(gw, "GET", "/website/mail/inbox") == 200
