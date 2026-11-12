"""A website browser's log: CDP methods and ops marked acting or reading; what could carry a secret taken out of an
entry, and a long value cut; a driver's tool call carried in a header; lines numbered in order, read in order up to
the last whose every earlier line is written, the sequence carried on by a server that restarts; and each tab event
credited to the command that caused it, to agentenv, or to the page after the latest action on its tab."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import types

import pytest

from agentenv_protocol.website_browser import log as browser_log
from agentenv_protocol.website_browser.log import (ACTOR, CUT_CHARS, REDACTED, BrowserLog, acts, caused, cut, op_acts, read_tool_call,
                                           redact_text, redact_url, redacted_args, redacted_op, redacted_params,
                                           tool_call_header)
from agentenv_protocol.computer.errors import InvalidArgument
from agentenv_protocol.website_browser.control import BrowserControl


@pytest.mark.parametrize("method, acting", [
    ("Input.dispatchMouseEvent", True), ("Input.insertText", True), ("Page.navigate", True), ("Page.reload", True),
    ("Runtime.evaluate", True), ("Runtime.callFunctionOn", True), ("DOM.setAttributeValue", True),
    ("Fetch.fulfillRequest", True), ("Network.setCookie", True), ("Storage.setCookies", True),
    ("Emulation.setDeviceMetricsOverride", True), ("Target.createTarget", True), ("Target.closeTarget", True),
    ("DOM.getDocument", False), ("DOMSnapshot.captureSnapshot", False), ("Accessibility.getFullAXTree", False),
    ("Page.captureScreenshot", False), ("Page.enable", False), ("Page.setLifecycleEventsEnabled", False),
    ("Network.getCookies", False), ("Target.getTargets", False), ("Target.attachToTarget", False),
    ("Runtime.getProperties", False), ("Browser.getVersion", False),
])
def test_a_cdp_method_acts_when_it_can_change_the_browser_or_a_page_and_reads_otherwise(method, acting):
    assert acts(method) is acting


def test_an_op_acts_when_it_can_change_the_browser_or_a_page():
    assert all(op_acts(op) for op in ("page/input", "navigate", "history", "tabs/open", "tabs/close", "cookies/set"))
    assert not any(op_acts(op) for op in ("page/capture", "page/state", "tabs/list", "cookies/get", "downloads/list"))


def test_a_url_keeps_its_search_terms_and_loses_its_credentials():
    assert redact_url("https://shop.test/search?q=standing+desk&keyword=oak&page=2") == \
        "https://shop.test/search?q=standing%20desk&keyword=oak&page=2"
    assert redact_url("https://app.test/cb?code=abc&state=s1&access_token=t#id_token=x&scope=email") == \
        f"https://app.test/cb?code={REDACTED}&state=s1&access_token={REDACTED}#id_token={REDACTED}&scope=email"
    assert redact_url("https://bucket.test/k?X-Amz-Signature=sig&X-Amz-Date=1&apiKey=k&sessionId=s") == \
        f"https://bucket.test/k?X-Amz-Signature={REDACTED}&X-Amz-Date={REDACTED}&apiKey={REDACTED}&sessionId={REDACTED}"
    assert redact_url("https://ada:hunter2@site.test/x#top") == f"https://ada:{REDACTED}@site.test/x#top"
    assert redact_url("about:blank") == "about:blank" and redact_url(None) is None


def test_cookie_header_and_body_values_are_taken_out_of_an_acting_entry():
    assert redacted_params("Network.setCookie", {"name": "sid", "value": "s3cret", "url": "https://a.test/?token=t"}) \
        == {"name": "sid", "value": REDACTED, "url": f"https://a.test/?token={REDACTED}"}
    assert redacted_params("Storage.setCookies", {"cookies": [{"name": "a", "value": "1"}]}) == \
        {"cookies": [{"name": "a", "value": REDACTED}]}
    assert redacted_params("Network.setCookies", {"cookies": [{"name": "a", "value": "1", "url": "https://a.test/?token=t"}]}) \
        == {"cookies": [{"name": "a", "value": REDACTED, "url": f"https://a.test/?token={REDACTED}"}]}
    fulfilled = redacted_params("Fetch.fulfillRequest", {"requestId": "r", "responseCode": 200, "body": "PGh0bWw+"})
    assert fulfilled["body"] == {"redacted": True, "length": 8, "sha256": hashlib.sha256(b"PGh0bWw+").hexdigest()}
    assert redacted_params("Page.setDocumentContent", {"frameId": "f", "html": "<p>faked</p>"})["html"]["redacted"]
    assert redacted_params("Network.setExtraHTTPHeaders", {"headers": {"Authorization": "Bearer x"}}) == \
        {"headers": {"Authorization": REDACTED}}
    continued = redacted_params("Fetch.continueRequest", {"requestId": "r", "postData": "cGFzcz0x",
                                                          "headers": [{"name": "Cookie", "value": "sid=1"}]})
    assert continued["headers"] == [{"name": "Cookie", "value": REDACTED}]
    assert continued["postData"] == {"redacted": True, "length": 8,
                                     "sha256": hashlib.sha256(b"cGFzcz0x").hexdigest()}
    assert redacted_params("Fetch.continueWithAuth", {"authChallengeResponse": {
        "response": "ProvideCredentials", "username": "ada", "password": "pw"}})["authChallengeResponse"] == {
        "response": "ProvideCredentials", "username": REDACTED, "password": REDACTED}


def test_typing_while_a_password_field_has_focus_keeps_only_how_much_was_typed():
    key = {"type": "keyDown", "key": "h", "code": "KeyH", "text": "h", "unmodifiedText": "h",
           "windowsVirtualKeyCode": 72, "modifiers": 0}
    assert redacted_params("Input.dispatchKeyEvent", key, password=True) == \
        {"type": "keyDown", "modifiers": 0, "text": {"length": 1}}
    assert redacted_params("Input.dispatchKeyEvent", key) == key
    assert redacted_params("Input.insertText", {"text": "hunter2"}, password=True) == {"text": {"length": 7}}
    events = [{"method": "Input.insertText", "params": {"text": "hunter2"}}]
    assert redacted_op("page/input", {"events": events}, password=True) == \
        {"events": [{"method": "Input.insertText", "params": {"text": {"length": 7}}}]}
    assert redacted_args("computer_type", {"text": "hunter2"}, password=True) == {"text": {"length": 7}}
    assert redacted_args("computer_type", {"text": "desk"}) == {"text": "desk"}
    assert redacted_args("computer_key", {"keys": ["h"]}, password=True) == {"keys": {"length": 1}}
    long_text = read_tool_call(tool_call_header("computer_type", "c", {"text": "p" * (CUT_CHARS + 5)}))["args"]
    assert redacted_args("computer_type", long_text, password=True) == {"text": {"length": CUT_CHARS + 5}}
    assert redacted_op("cookies/set", {"cookies": [{"name": "a", "value": "1"}]}) == \
        {"cookies": [{"name": "a", "value": REDACTED}]}


def test_playwright_mcps_typing_arguments_keep_only_their_length_while_a_password_field_has_focus():
    fields = {"fields": [{"name": "Email", "type": "textbox", "value": "ada@x.test"},
                         {"name": "Password", "type": "textbox", "value": "hunter2"}, {"name": "Remember"}]}
    assert redacted_args("browser_type", {"target": "e3", "text": "hunter2"}, password=True) == \
        {"target": "e3", "text": {"length": 7}}
    assert redacted_args("browser_press_key", {"key": "h"}, password=True) == {"key": {"length": 1}}
    assert redacted_args("browser_fill_form", fields, password=True) == {"fields": [
        {"name": "Email", "type": "textbox", "value": {"length": 10}},
        {"name": "Password", "type": "textbox", "value": {"length": 7}}, {"name": "Remember"}]}
    assert redacted_args("browser_fill_form", fields) == fields


def test_a_tab_event_a_tool_calls_command_caused_names_the_call(tmp_path):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    command = {"seq": 7, "via": "cdp", "who": "playwright-mcp"}

    assert caused({**command, "for": 5}, log, "T1") == {"by": "command", "entry": 7, "via": "cdp",
                                                         "who": "playwright-mcp", "for": 5}
    assert caused({**command, "for": [5, 6]}, log, "T1")["for"] == [5, 6]
    assert "for" not in caused(command, log, "T1")


def test_a_long_value_is_cut_with_its_length_and_sha256():
    script = "x" * (CUT_CHARS + 10)
    assert redacted_params("Runtime.evaluate", {"expression": script})["expression"] == {
        "cut": "x" * CUT_CHARS, "length": CUT_CHARS + 10, "sha256": hashlib.sha256(script.encode()).hexdigest()}
    assert cut({"a": ("b", types.MappingProxyType({"c": 1}))}) == {"a": ["b", {"c": 1}]}


def test_a_tool_call_header_carries_the_tool_its_call_and_its_arguments_cut():
    args = types.MappingProxyType({"text": "y" * (CUT_CHARS + 1), "x": 3})
    read = read_tool_call(tool_call_header("computer_type", "c1", args))
    assert read == {"tool": "computer_type", "call": "c1", "args": {"text": cut("y" * (CUT_CHARS + 1)), "x": 3}}
    assert read_tool_call(None) is None and read_tool_call("not base64 json") is None
    assert read_tool_call(base64.urlsafe_b64encode(b'{"call": 1}').decode()) is None


def test_a_long_url_and_a_url_in_an_error_keep_no_credentials():
    long_url = "https://a.test/?token=secret&pad=" + "x" * CUT_CHARS
    sent = read_tool_call(tool_call_header("computer_navigate", "c", {"url": long_url}))["args"]["url"]
    assert sent["cut"].startswith(f"https://a.test/?token={REDACTED}&pad=") and "secret" not in json.dumps(sent)
    kept = redacted_args("computer_navigate", {"url": cut(long_url)})["url"]
    assert kept["cut"].startswith(f"https://a.test/?token={REDACTED}") and "secret" not in json.dumps(kept)
    assert redact_text("the browser couldn't open https://a.test/?token=t: net::ERR_FAILED") == \
        f"the browser couldn't open https://a.test/?token={REDACTED}: net::ERR_FAILED"


def test_a_tool_call_header_too_long_even_cut_drops_its_arguments(monkeypatch):
    monkeypatch.setattr(browser_log, "HEADER_CHARS", 200)
    read = read_tool_call(tool_call_header("computer_type", "c1", {"text": "z" * 500}))
    assert read["tool"] == "computer_type" and read["args"] == {}


@pytest.mark.asyncio
async def test_lines_are_numbered_as_they_begin_and_read_in_order_up_to_the_first_unwritten(tmp_path):
    log = BrowserLog(tmp_path / "logs" / "browser-chromium.jsonl")
    first = log.begin({"via": "cdp", "method": "Page.navigate", "acts": True, "tab": "T1"})
    second = log.note("tab", {"event": "navigated", "id": "T1"})
    assert log.read() == {"lines": [], "last": 0, "more": False}
    log.finish(first, ok=True)
    read = log.read()
    assert [line["seq"] for line in read["lines"]] == [1, 2] and read["last"] == second
    assert read["lines"][0]["action"] == {"via": "cdp", "method": "Page.navigate", "acts": True, "tab": "T1",
                                          "ok": True}
    assert log.read(after=1)["lines"][0]["tab"] == {"event": "navigated", "id": "T1"}
    assert log.last_acting == {"T1": 1}
    assert log.read(after=1, limit=0) == {"lines": [], "last": 1, "more": True}
    assert (await log.complete_text()).count("\n") == 2
    assert await log.read_async(after=1) == log.read(after=1)


def test_an_action_still_going_after_the_limit_is_written_unfinished_and_its_outcome_after(tmp_path, monkeypatch):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    slow = log.begin({"via": "cdp", "method": "Runtime.evaluate", "acts": True, "tab": "T1"})
    log.note("reset", {})
    assert log.read()["lines"] == []
    monkeypatch.setattr(browser_log, "STALLED_S", 0)
    first = log.read()
    assert [line["seq"] for line in first["lines"]] == [slow["seq"], slow["seq"] + 1] and first["last"] == 2
    assert first["lines"][0]["action"] == {"via": "cdp", "method": "Runtime.evaluate", "acts": True, "tab": "T1",
                                           "unfinished": True}
    log.finish(slow, ok=True)
    later = log.read(after=first["last"])
    assert later["lines"] == [{"seq": 3, "at": later["lines"][0]["at"], "outcome": {"entry": slow["seq"], "ok": True}}]
    assert [line["seq"] for line in log.read()["lines"]] == [1, 2, 3]


def test_a_read_settling_through_an_entry_writes_the_actions_still_going_up_to_it_unfinished_now(tmp_path):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    early = log.begin({"via": "mcp", "tool": "browser_navigate", "acts": True, "tab": "T1"})
    late = log.begin({"via": "cdp", "method": "Runtime.evaluate", "acts": True, "tab": "T1"})

    read = log.read(settle_through=early["seq"])

    assert [(line["seq"], line["action"]["unfinished"]) for line in read["lines"]] == [(early["seq"], True)]
    assert read["last"] == early["seq"] and log.read(after=early["seq"])["lines"] == []
    log.finish(late, ok=True)
    assert [line["seq"] for line in log.read(after=early["seq"])["lines"]] == [late["seq"]]


def test_a_restarted_server_carries_on_the_sequence_on_a_new_line_after_a_partial_one(tmp_path):
    path = tmp_path / "browser-chromium.jsonl"
    path.write_text('{"seq": 1, "at": "t", "reset": {}}\n{"seq": 2, "at": "t", "tab": {}}\n{"seq": 3, "at"')
    log = BrowserLog(path)
    log.note("reset", {})
    assert [line["seq"] for line in log.read()["lines"]] == [1, 2, 3]
    assert path.read_text().endswith('\n{"seq": 3, "at"\n' + json.dumps(log.read()["lines"][-1]) + "\n")


def test_a_log_that_cant_be_written_warns_and_keeps_its_numbers(tmp_path, caplog):
    blocked = tmp_path / "file"
    blocked.write_text("")
    log = BrowserLog(blocked / "browser-chromium.jsonl")
    log.note("reset", {})
    log.note("reset", {})
    assert log.read() == {"lines": [], "last": 2, "more": False}
    assert len([r for r in caplog.records if "can't be written" in r.getMessage()]) == 1


class Pipe:
    """A browser's pipe for agentenv's control: a new tab's target said to open before it's answered, a navigation
    said to commit before its loader is answered, a close said to destroy its target."""

    def __init__(self) -> None:
        self.listeners, self.tabs, self.focus, self.checked = [], 0, "text", []

    def listen(self, listener) -> None:
        self.listeners.append(listener)

    def emit(self, method: str, params: dict, session=None) -> None:
        for listener in self.listeners:
            listener({"method": method, "params": params, **({"sessionId": session} if session else {})})

    def open(self, target: str, opener=None) -> None:
        self.emit("Target.targetCreated", {"targetInfo": {"targetId": target, "type": "page", "url": "about:blank",
                                                          **({"openerId": opener} if opener else {})}})
        self.emit("Target.attachedToTarget", {"sessionId": f"s-{target}", "targetInfo": {"targetId": target,
                                                                                          "type": "page"}})

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method == "Target.createTarget":
            self.tabs += 1
            self.open(target := f"N{self.tabs}")
            return {"targetId": target}
        if method == "Page.navigate":
            target = session_id.removeprefix("s-")
            self.emit("Page.frameNavigated", {"frame": {"id": target, "url": params["url"], "loaderId": "L-nav"}},
                      session_id)
            return {"frameId": target, "loaderId": "L-nav"}
        if method == "Target.closeTarget":
            self.emit("Target.targetDestroyed", {"targetId": params["targetId"]})
            return {"success": True}
        if method == "Runtime.evaluate" and "visibilityState" in params["expression"]:
            return {"result": {"value": ["visible", True]}}
        if method == "Runtime.evaluate" and "password" in params["expression"]:
            self.checked.append(self.focus == "password")
            return {"result": {"value": self.focus == "password"}}
        if method == "Input.dispatchKeyEvent" and params.get("key") == "Tab":
            self.focus = "password"
        if method == "Page.getNavigationHistory":
            return {"currentIndex": 0, "entries": [{"id": 1, "url": "about:blank"}]}
        return {}

    async def request(self, method, params=None, *, session_id=None, answered=None):
        return {"result": await self.call(method, params, session_id=session_id)}


async def _settled() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_move_within_a_document_is_the_commands_that_asked_for_it_and_uses_up_a_history_move():
    causes = browser_log.Causes()
    asked, back = {"seq": 3, "via": "cdp", "who": "ada"}, {"seq": 4, "via": "mcp", "who": "bo"}
    answer = asyncio.get_running_loop().create_future()
    causes.navigating("T1", answer, asked)
    answer.set_result(None)
    assert await causes.navigated_within("T1") == asked
    causes.navigates_next("T1", back)
    assert await causes.navigated_within("T1") == back
    assert await causes.navigated("T1", "L-page") is None
    failed = asyncio.get_running_loop().create_future()
    causes.navigating("T1", failed, asked)
    failed.cancel()
    await asyncio.sleep(0)
    assert await causes.navigated_within("T1") is None


@pytest.mark.asyncio
async def test_a_move_back_with_nowhere_to_go_leaves_no_cause_behind(tmp_path):
    pipe = Pipe()
    control = BrowserControl(pipe, "agent", (1280, 800))
    control.log = log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    pipe.open("T1")
    await _settled()
    token = ACTOR.set({"seq": 99, "via": "mcp", "who": "bo", "acts": True, "tab": None, "password": None})
    try:
        with pytest.raises(InvalidArgument):
            await control.history("back")
    finally:
        ACTOR.reset(token)
    pipe.emit("Page.frameNavigated", {"frame": {"id": "T1", "url": "https://a.test/", "loaderId": "L-page"}}, "s-T1")
    await _settled()
    assert [line["tab"] for line in log.read()["lines"] if line["tab"]["event"] == "navigated"] == [
        {"event": "navigated", "id": "T1", "url": "https://a.test/", "by": "page", "after": 99}]


@pytest.mark.asyncio
async def test_typing_that_tabs_into_a_password_field_is_withheld_from_then_on(tmp_path):
    pipe = Pipe()
    control = BrowserControl(pipe, "agent", (1280, 800))
    control.log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    pipe.open("T1")
    await _settled()
    actor = {"seq": 1, "via": "mcp", "who": "bo", "acts": True, "tab": None, "password": None}
    token = ACTOR.set(actor)
    try:
        await control.dispatch([{"method": "Input.dispatchKeyEvent", "params": {"type": "keyDown", "key": "a", "text": "a"}},
                                {"method": "Input.dispatchKeyEvent", "params": {"type": "keyDown", "key": "Tab"}},
                                {"method": "Input.dispatchKeyEvent", "params": {"type": "keyDown", "key": "s", "text": "s"}}])
    finally:
        ACTOR.reset(token)
    assert actor["password"] is True and actor["tab"] == "T1"
    assert pipe.checked == [False, False, True]


@pytest.mark.asyncio
async def test_each_tab_event_names_what_caused_it(tmp_path):
    pipe = Pipe()
    control = BrowserControl(pipe, "agent", (1280, 800))
    control.log = log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    control._launching = True
    pipe.open("T0")
    control._launching = False
    await _settled()

    begun = log.begin({"via": "mcp", "who": "ada", "op": "tabs/open", "acts": True})
    token = ACTOR.set({"seq": begun["seq"], "via": "mcp", "who": "ada", "acts": True, "tab": None})
    try:
        opened = await control.open_tab("https://shop.test/?q=desk&token=t")
    finally:
        ACTOR.reset(token)
    log.finish(begun, ok=True)
    await _settled()
    clicked = log.begin({"via": "cdp", "who": "bo", "method": "Input.dispatchMouseEvent", "acts": True, "tab": opened})
    log.finish(clicked, ok=True)
    pipe.open("P1", opener=opened)
    pipe.emit("Page.frameNavigated", {"frame": {"id": opened, "url": "https://shop.test/item", "loaderId": "L-page"}},
              f"s-{opened}")
    pipe.emit("Page.navigatedWithinDocument", {"frameId": opened, "url": "https://shop.test/item#reviews"},
              f"s-{opened}")
    await control.close_tab("P1")
    control.ended("stop")
    await _settled()

    events = [line["tab"] for line in log.read()["lines"] if "tab" in line]
    assert events == [
        {"event": "opened", "id": "T0", "url": "about:blank", "by": "agentenv"},
        {"event": "opened", "id": opened, "url": "about:blank", "by": "command", "entry": begun["seq"], "via": "mcp",
         "who": "ada"},
        {"event": "navigated", "id": opened, "url": f"https://shop.test/?q=desk&token={REDACTED}", "by": "command",
         "entry": begun["seq"], "via": "mcp", "who": "ada"},
        {"event": "opened", "id": "P1", "url": "about:blank", "opener": opened, "by": "page", "after": clicked["seq"]},
        {"event": "navigated", "id": opened, "url": "https://shop.test/item", "by": "page", "after": clicked["seq"]},
        {"event": "navigated", "id": opened, "url": "https://shop.test/item#reviews", "within_document": True,
         "by": "page", "after": clicked["seq"]},
        {"event": "closed", "id": "P1", "url": "about:blank", "by": "agentenv"},
        {"event": "closed", "id": "T0", "url": "about:blank", "by": "stop"},
        {"event": "closed", "id": opened, "url": "about:blank", "by": "stop"},
    ]
