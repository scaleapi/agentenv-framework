"""Coordinate spaces: frame pixels and a 0..norm_scale grid, one mapping to the frame's pixels, refused and never
clamped outside the frame or the range, with the default space set per deploy."""
from __future__ import annotations

import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")
fastmcp = pytest.importorskip("fastmcp")

from computer_fake import FakeDriver, FakeScreen, http_client, mcp_session  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol import EnvironmentTool  # noqa: E402
from agentenv_protocol.computer import ComputerSettings, InvalidArgument, OutOfBounds  # noqa: E402
from agentenv_protocol.computer.frames import FrameStore  # noqa: E402
from agentenv_protocol.computer.operations import Display  # noqa: E402
from agentenv_protocol.computer.spaces import check_space, frame_point  # noqa: E402


def _frame(width: int, height: int):
    return FrameStore().add(PILImage.new("RGB", (width, height)), Display(width, height), (0, {}), identity="default")


@pytest.mark.parametrize("width, height, x, y, expected", [
    (1280, 800, 0, 0, (0, 0)),
    (1280, 800, 1000, 1000, (1279, 799)),
    (1280, 800, 500, 500, (640, 400)),
    (1279, 799, 500, 500, (639.5, 399.5)),
    (3, 3, 999, 0, (2, 0)),
    (3, 3, 666, 334, (1.998, 1.002)),
])
def test_norm_points_land_on_the_frames_pixels(width, height, x, y, expected):
    fx, fy = frame_point(_frame(width, height), x, y, "norm", 1000)
    assert (round(fx, 3), round(fy, 3)) == expected


def test_frame_points_are_its_pixels_and_odd_pixels_stay_exact():
    frame = _frame(1279, 799)
    assert frame_point(frame, 1278, 798, "frame", 1000) == (1278, 798)
    assert frame.to_screen(1278.9, 0.5) == (1278, 0)


@pytest.mark.parametrize("x, y, space", [(1280, 0, "frame"), (0, -1, "frame"), (1000.5, 0, "norm"), (-0.1, 5, "norm")])
def test_points_outside_the_frame_or_range_are_refused(x, y, space):
    with pytest.raises(OutOfBounds):
        frame_point(_frame(1280, 800), x, y, space, 1000)


@pytest.mark.parametrize("space, scale", [("grid", None), (None, 0), (None, 2_000_000), (None, 1.5)])
def test_a_bad_space_or_scale_is_refused(space, scale):
    with pytest.raises(InvalidArgument):
        check_space(space, scale, "frame", 1000)


@pytest.mark.asyncio
async def test_the_default_space_and_scale_come_from_the_settings_and_a_call_can_override_them():
    driver = FakeDriver(FakeScreen(1280, 800), settings=ComputerSettings(default_space="norm", default_norm_scale=999))
    async with mcp_session(driver) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        await session.call_tool("computer_screenshot", {})
        await session.call_tool("computer_click", {"x": 999, "y": 0})
        await session.call_tool("computer_click", {"x": 7, "y": 9, "space": "frame"})
        await session.call_tool("computer_click", {"x": 1, "y": 1, "norm_scale": 1})
    assert "0 to 999 over the frame" in tools["computer_click"].description
    assert "0 to 999 over the frame" in tools["computer_drag"].description
    assert "default space" not in tools["computer_type"].description
    assert [entry[1:3] for entry in driver.screen.log] == [(1279, 0), (7, 9), (1279, 799)]


@pytest.mark.asyncio
async def test_setting_the_default_space_rewrites_the_descriptions():
    driver = FakeDriver()
    async with http_client(driver) as http, mcp_session(driver) as session:
        before = {t.name: t.description for t in (await session.list_tools()).tools}
        applied = (await http.post("/agentenv/ext/computer-spaces/set",
                                   json={"default_space": "norm", "default_norm_scale": 1000})).json()
        after = {t.name: t.description for t in (await session.list_tools()).tools}
        card = (await http.get("/.well-known/agent-env.json")).json()
        current = (await http.get("/agentenv/ext/computer-spaces/get")).json()
        refused = await http.post("/agentenv/ext/computer-spaces/set", json={"default_space": "grid"})
    assert "frame space by default" in before["computer_click"]
    assert applied == current == {"default_space": "norm", "default_norm_scale": 1000}
    assert "norm space by default: 0 to 1000" in after["computer_click"]
    card_click = next(t for t in card["capabilities"]["tools"] if t["name"] == "computer_click")
    assert card_click["description"] == after["computer_click"]
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_setting_the_default_space_rewrites_the_descriptions_on_fastmcp():
    app = fastmcp.FastMCP("desktop")
    FakeDriver().mount(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.http_app()), base_url="http://t") as http:
        await http.post("/agentenv/ext/computer-spaces/set", json={"default_space": "norm", "default_norm_scale": 1000})
        card = (await http.get("/.well-known/agent-env.json")).json()
    async with fastmcp.Client(app) as client:
        listed = {t.name: t for t in await client.list_tools()}
        await client.call_tool("computer_screenshot", {})
        clicked = await client.call_tool("computer_click", {"x": 500, "y": 500})
    card_click = next(t for t in card["capabilities"]["tools"] if t["name"] == "computer_click")
    assert "norm space by default: 0 to 1000" in listed["computer_click"].description
    assert card_click["description"] == listed["computer_click"].description
    assert clicked.structured_content["summary"].startswith("Clicked left at (500, 500) (norm 1000) in f1")


@pytest.mark.asyncio
async def test_setting_the_default_space_rewrites_only_the_point_tools_the_driver_serves():
    class NoScroll(FakeDriver):
        def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
            return [t for t in super().served_tools(tools) if t.name != "computer_scroll"]

    driver = NoScroll()
    async with http_client(driver) as http, mcp_session(driver) as session:
        applied = await http.post("/agentenv/ext/computer-spaces/set", json={"default_space": "norm"})
        listed = {t.name: t.description for t in (await session.list_tools()).tools}
    assert applied.status_code == 200 and "computer_scroll" not in listed
    assert "norm space by default" in listed["computer_click"] and "norm space by default" in listed["computer_drag"]
