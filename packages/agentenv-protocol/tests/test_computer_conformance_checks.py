"""Each probe check against stubs that answer the way a driver outside our base class might: the failure each way is
reported with what was seen, and checks for what a driver doesn't declare are left out."""
from __future__ import annotations

import base64
import contextlib
import io
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
pytest.importorskip("PIL")

from mcp.types import CallToolResult, ImageContent, TextContent, Tool  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol.client import ExtensionCallError  # noqa: E402
from agentenv_protocol.computer.card import (  # noqa: E402
    COMPUTER,
    COMPUTER_EXEC,
    COMPUTER_POINTER,
    COMPUTER_READY,
    COMPUTER_SPACES,
    GROUP_TOOLS,
)
from agentenv_protocol.computer.conformance import checks  # noqa: E402
from agentenv_protocol.tool_access import DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI  # noqa: E402
from agentenv_protocol.types import INTAKE_EXTENSION_URI  # noqa: E402

SCHEMA = {"type": "object"}


def png(width: int, height: int, colour=(255, 255, 255)) -> str:
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), colour).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def shot(width=100, height=80, frame="f1", parent=None, image=None, display=None) -> CallToolResult:
    data = {"frame_id": frame, "width": width, "height": height, "parent_frame_id": parent,
            "display": dict(zip(("width", "height", "scale"), display or (width, height, 1.0)))}
    return CallToolResult(content=[TextContent(type="text", text="shot"),
                                   ImageContent(type="image", data=image or png(width, height), mimeType="image/png")],
                          structuredContent=data)


def ok(**data) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text="ok")], structuredContent=data)


def error(text: str, **data) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], structuredContent=data, isError=True)


class Session:
    def __init__(self, call=None, tools=None) -> None:
        self.call = call or (lambda name, args: shot() if name == "computer_screenshot" else ok())
        self.tools = tools if tools is not None else [*checks.MINIMUM_TOOLS]

    async def list_tools(self):
        return SimpleNamespace(tools=[Tool(name=n, inputSchema=SCHEMA, outputSchema=SCHEMA) for n in self.tools])

    async def call_tool(self, name, args=None):
        return self.call(name, args or {})


class Driver:
    def __init__(self, extensions=(), params=None, invoke=None, run=None, rpc=None, other=None) -> None:
        self.extensions = list(extensions)
        self._params = params or {}
        self._invoke = invoke or (lambda uri, method, params: {})
        self._run = run or (lambda argv: {"exit_code": 0, "stdout": "", "stderr": ""})
        self._rpc = rpc or (lambda method, params: {"result": {}})
        self.other = other or Session()
        self.card = {"capabilities": {"extensions": [{"uri": u, "params": self._params.get(u, {})}
                                                     for u in self.extensions]}}

    def declares(self, uri):
        return uri in self.extensions

    def params(self, uri):
        return self._params.get(uri, {})

    async def invoke(self, uri, method, params=None):
        return self._invoke(uri, method, params or {})

    async def run(self, *argv, timeout_s=30):
        return self._run(list(argv))

    async def rpc(self, method, params=None):
        return self._rpc(method, params or {})

    async def add_file(self, path, data):
        return self._rpc("data/add", {"path": path})

    @contextlib.asynccontextmanager
    async def session(self, role=None):
        yield self.other


def probe(driver=None, session=None) -> checks.Probe:
    return checks.Probe(driver or Driver([COMPUTER]), session or Session(), 0.1)


def details(p: checks.Probe) -> dict[str, tuple[str, str]]:
    return {r.name: (r.status, r.detail) for r in p.results}


@pytest.mark.asyncio
async def test_a_card_without_readiness_or_tools_fails_p0_and_p1():
    p = probe(Driver([COMPUTER, COMPUTER_POINTER]), Session(tools=["computer_screenshot", "computer_pointer_move"]))
    await p._ready()
    await p._card()
    got = details(p)
    assert got["computer-ready reports ready"] == ("failed", "the card doesn't declare computer-ready/v1")
    assert got["the seven minimum tools are listed with their schemas"][1].startswith("missing ['computer_click'")
    assert "computer-pointer/v1 is declared without" in got["each group's tools come with its extension"][1]


@pytest.mark.asyncio
async def test_group_tools_without_their_extension_fail_p1():
    p = probe(session=Session(tools=[*checks.MINIMUM_TOOLS, *GROUP_TOOLS["keyboard"]]))
    await p._card()
    assert "are listed without urn:agentenv:computer-keyboard/v1" in details(p)["each group's tools come with its extension"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, name, problem", [
    (lambda n, a: shot(display=(50, 40, 1.0)), "a screenshot is native: its image, frame and display agree",
     "the display (50, 40)"),
    (lambda n, a: error("out_of_bounds: no") if a.get("region") else shot(), "a region comes back as a native crop of its frame",
     "out_of_bounds: no"),
    (lambda n, a: shot(10, 10, frame="f2", parent="f1") if a.get("region") else shot(),
     "a region comes back as a native crop of its frame", "region came back (10, 10)"),
    (lambda n, a: shot(33, 26, frame="f2", parent="f9") if a.get("region") else shot(),
     "a region comes back as a native crop of its frame", "its parent_frame_id is 'f9'"),
    (lambda n, a: shot(33, 26, frame="f2", parent="f1", image=png(33, 26, (0, 0, 0))) if a.get("region") else shot(),
     "a region comes back as a native crop of its frame", "aren't its frame's"),
])
async def test_frames_that_arent_native_fail_p2(answer, name, problem):
    p = probe(session=Session(call=answer))
    await p._frame()
    status, detail = details(p)[name]
    assert status == "failed" and problem in detail


@pytest.mark.asyncio
async def test_extensions_a_driver_doesnt_declare_get_no_checks():
    p = probe()
    for section in (p._display, p._spaces, p._keyboard_settings, p._exec, p._data_get, p._tool_access, p._pointer,
                    p._pointer_input):
        await section()
    assert p.results == []


@pytest.mark.asyncio
async def test_a_spaces_set_that_doesnt_take_fails():
    driver = Driver([COMPUTER, COMPUTER_SPACES], invoke=lambda uri, method, params: {
        "default_space": "frame", "default_norm_scale": 1000})
    p = probe(driver)
    await p._spaces()
    assert "set norm, then get said" in next(iter(details(p).values()))[1]


@pytest.mark.asyncio
async def test_exec_that_runs_a_missing_program_fails_and_no_cap_means_no_cap_check():
    def invoke(uri, method, params):
        if params["argv"] == ["agentenv-probe-no-such-program"]:
            return {"exit_code": 127, "stdout": "", "stderr": "not found"}
        raise ExtensionCallError(SimpleNamespace(status_code=400, json=lambda: {"error": {}}, text="",
                                                 request=None))

    p = probe(Driver([COMPUTER, COMPUTER_EXEC], invoke=invoke))
    p.driver._run = lambda argv: invoke(None, None, {"argv": argv})
    await p._exec()
    got = details(p)
    assert got["a program that isn't there is refused with a code"] == ("failed", "it answered {'exit_code': 127, "
                                                                        "'stdout': '', 'stderr': 'not found'}")
    assert "output past max_output_bytes is cut there" not in got


@pytest.mark.asyncio
@pytest.mark.parametrize("listed, refused, problem", [
    (["computer_wait"], error("x", code="permission_denied"), "still listed"),
    ([], ok(), "a disabled tool answered its role"),
])
async def test_tool_access_that_doesnt_hide_or_refuse_fails(listed, refused, problem):
    other = Session(call=lambda n, a: refused, tools=listed)
    p = probe(Driver([COMPUTER, DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI], other=other))
    await p._tool_access()
    assert problem in next(iter(details(p).values()))[1]


@pytest.mark.asyncio
async def test_a_fixture_data_add_refuses_is_unverified_and_a_root_user_fails_p9():
    refusing = Driver([COMPUTER, COMPUTER_EXEC, INTAKE_EXTENSION_URI],
                      rpc=lambda method, params: {"error": {"message": "read-only home"}})
    refusing.card["capabilities"]["extensions"][-1]["params"] = {"add": [{}]}
    p = probe(refusing)
    await p.full()
    assert {r.detail for r in p.results} == {"the fixture didn't start: data/add refused it: read-only home"}

    root = probe(Driver([COMPUTER, COMPUTER_EXEC], run=lambda argv: {"stdout": "0\n" if argv == ["id", "-u"] else "0\n"}))
    await root._security()
    assert details(root)["commands and data/add run as one unprivileged user"] == ("failed", "commands run as root")
    await root._unstage()


def fixture(state: dict, layout: dict | None = None):
    """A driver and probe whose fixture is up with ``layout`` and answers ``state``."""
    run = lambda argv: {"exit_code": 0, "stdout": json.dumps(state)}  # noqa: E731
    p = probe(Driver([COMPUTER, COMPUTER_EXEC], run=run))
    p.layout = layout or {"grid": [0, 0, 80, 80], "save": [0, 0, 10, 10], "subscribe": [0, 0, 10, 10],
                          "pad": [0, 0, 80, 60], "name": [0, 0, 10, 10], "markers": [[5, 5, "#ffffff"]], "cell": 10}
    return p


@pytest.mark.asyncio
async def test_effects_without_spaces_right_or_double_clicks_leave_those_out_and_report_refusals(monkeypatch):
    monkeypatch.setattr(checks, "SETTLE_S", 0.01)
    p = fixture({"cells": {}, "right_clicks": 0, "double_clicks": 0, "saved_count": 0, "subscribed": False})
    p.session.call = lambda n, a: (shot() if n == "computer_screenshot" else error("busy") if a.get("x") else ok())
    p.features = {"buttons": ["left"], "max_click_count": 1}
    await p._geometry()
    await p._effects()
    got = details(p)
    assert "a click in norm space hits its cell" not in got and "a right click arrives as one" not in got
    assert "a double click arrives as one" not in got and "norm points land on the same pixels" not in got
    assert got["16 clicks sent at once all land"] == ("failed", "16 of 16 were refused")
    assert got["a click in frame space hits its cell, and the next frame shows it"][1].startswith("AssertionError")


@pytest.mark.asyncio
async def test_crops_the_driver_refuses_fail_p3_and_p4(monkeypatch):
    monkeypatch.setattr(checks, "SETTLE_S", 0.01)
    p = fixture({"cells": {}})
    p.driver.extensions.append(COMPUTER_SPACES)
    p.session.call = lambda n, a: error("no crops") if a.get("region") else shot()
    await p._geometry()
    await p._effects()
    got = details(p)
    assert got["norm points land on the same pixels"] == ("failed", "no crops")
    assert got["a click in a crop's frame maps back to the screen"] == ("failed", "no crops")


@pytest.mark.asyncio
@pytest.mark.parametrize("drags, problem", [
    ([], "no drag reached the pad"),
    ([{"start": [0, 0], "end": [300, 250], "points": 9, "path": []}], "the stroke was [0, 0] to [300, 250] in 9 points"),
])
async def test_a_drag_that_doesnt_arrive_whole_fails_p7(drags, problem, monkeypatch):
    monkeypatch.setattr(checks, "SETTLE_S", 0.01)
    p = fixture({"drags": drags, "scroll_top": 0, "wheel": {}})
    await p._scroll_and_drag()
    assert details(p)["a drag starts and ends where it says, through its waypoints"] == ("failed", problem)


@pytest.mark.asyncio
async def test_typing_without_the_keyboard_group_leaves_held_keys_out(monkeypatch):
    monkeypatch.setattr(checks, "SETTLE_S", 0.01)
    p = fixture({"name": "", "notes": "", "keys": [], "clipboard": "x"}, {"name": [0, 0, 10, 10],
                                                                           "notes": [0, 0, 10, 10], "clipboard": "x"})
    await p._typing()
    assert "a key held by computer_keyboard_down stays down while text is typed" not in details(p)


@pytest.mark.asyncio
async def test_a_data_add_the_app_cant_see_fails():
    p = fixture({"loaded": None})
    p.driver._rpc = lambda method, params: {"error": {"message": "no inbox"}}
    await p._data_add()
    assert details(p)["a file data/add writes is there for an app to read"] == ("failed", "data/add refused it: no inbox")


@pytest.mark.asyncio
async def test_a_section_that_raises_outside_its_checks_fails_as_one():
    p = probe(session=Session(call=lambda n, a: error("down")))
    p.session.list_tools = None
    await p._guarded("P1", p._card)
    assert details(p)["card could run"][0] == "failed"
    assert COMPUTER_READY not in p.results[0].extensions
