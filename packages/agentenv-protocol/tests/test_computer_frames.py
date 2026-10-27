"""Frames: screenshots as frames by id, native crops that map back exactly, encodings, retention, staleness after a
display change, HiDPI mapping, and each caller's latest frame."""
from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver, FakeScreen, http_client, mcp_session  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol import ROLE_META_KEY  # noqa: E402
from agentenv_protocol.computer.frames import KEEP  # noqa: E402


def _image(result) -> PILImage.Image:
    return PILImage.open(io.BytesIO(base64.b64decode(result.content[1].data)))


@pytest.mark.asyncio
async def test_a_screenshot_is_a_native_frame_with_its_geometry():
    async with mcp_session(FakeDriver(FakeScreen(1280, 800))) as session:
        first = await session.call_tool("computer_screenshot", {})
        second = await session.call_tool("computer_screenshot", {"format": "jpeg", "quality": 40})
    data = first.structuredContent
    assert data["frame_id"] == "f1" and (data["width"], data["height"]) == (1280, 800)
    assert data["display"] == {"width": 1280, "height": 800, "scale": 1.0} and "parent_frame_id" not in data
    assert data["captured_at"].endswith("Z") and data["summary"] == "Captured f1: 1280×800 of the 1280×800 display"
    assert first.content[0].text == data["summary"] and first.content[1].mimeType == "image/png"
    assert _image(first).size == (1280, 800)
    assert second.structuredContent["frame_id"] == "f2" and second.content[1].mimeType == "image/jpeg"
    assert _image(second).format == "JPEG"


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{"quality": 50}, {"format": "jpeg", "quality": 0}, {"format": "jpeg", "quality": 96}])
async def test_quality_is_for_jpeg_within_range(args):
    async with mcp_session(FakeDriver()) as session:
        refused = await session.call_tool("computer_screenshot", args)
    assert refused.structuredContent["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_a_region_is_a_native_crop_whose_points_map_back_exactly():
    driver = FakeDriver(FakeScreen(1280, 800))
    driver.screen.paint(700, 450, (1, 2, 3))
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        crop = await session.call_tool("computer_screenshot", {"region": [600, 400, 900, 500]})
        nested = await session.call_tool("computer_screenshot", {"region": [100, 50, 200, 100], "frame_id": "f2"})
        clicked = await session.call_tool("computer_click", {"x": 1, "y": 1, "frame_id": "f3"})
    assert crop.structuredContent["parent_frame_id"] == "f1" and (crop.structuredContent["width"],
                                                                   crop.structuredContent["height"]) == (300, 100)
    assert crop.structuredContent["summary"] == "Cropped f2: 300×100 of f1 at (600, 400)"
    assert crop.structuredContent["captured_at"] == (await _captured_at(driver, "f1"))
    assert _image(crop).getpixel((100, 50))[:3] == (1, 2, 3)
    assert nested.structuredContent["parent_frame_id"] == "f2" and _image(nested).getpixel((0, 0))[:3] == (1, 2, 3)
    assert not clicked.isError and driver.screen.log[-1][1:3] == (701, 451)


async def _captured_at(driver: FakeDriver, frame_id: str) -> str:
    return driver._computer.frames.get(frame_id, "default").result().captured_at


@pytest.mark.asyncio
@pytest.mark.parametrize("args, code", [
    ({"frame_id": "f1"}, "invalid_argument"),
    ({"region": [0, 0, 10]}, "invalid_argument"),
    ({"region": [10, 10, 10, 20]}, "out_of_bounds"),
    ({"region": [0, 0, 1281, 10]}, "out_of_bounds"),
    ({"region": [-1, 0, 10, 10]}, "out_of_bounds"),
    ({"region": [0, 0, 10, 10], "frame_id": "f9"}, "stale_frame"),
    ({"region": [0, 0, 1001, 10], "space": "norm"}, "out_of_bounds"),
])
async def test_a_bad_region_is_refused(args, code):
    async with mcp_session(FakeDriver()) as session:
        await session.call_tool("computer_screenshot", {})
        refused = await session.call_tool("computer_screenshot", args)
    assert refused.isError and refused.structuredContent["code"] == code


@pytest.mark.asyncio
async def test_a_region_in_norm_space_crops_the_same_box():
    async with mcp_session(FakeDriver(FakeScreen(1000, 500))) as session:
        await session.call_tool("computer_screenshot", {})
        crop = await session.call_tool("computer_screenshot", {"region": [500, 500, 1000, 1000], "space": "norm"})
    assert (crop.structuredContent["width"], crop.structuredContent["height"]) == (500, 250)


@pytest.mark.asyncio
async def test_only_the_last_frames_are_kept():
    async with mcp_session(FakeDriver(FakeScreen(8, 8))) as session:
        for _ in range(KEEP + 1):
            await session.call_tool("computer_screenshot", {})
        evicted = await session.call_tool("computer_click", {"x": 1, "y": 1, "frame_id": "f1"})
        kept = await session.call_tool("computer_click", {"x": 1, "y": 1, "frame_id": "f2"})
    assert evicted.structuredContent["code"] == "stale_frame" and "no longer kept" in evicted.structuredContent["message"]
    assert evicted.structuredContent["retryable"] is True and not kept.isError


@pytest.mark.asyncio
async def test_a_frame_goes_stale_when_the_display_changes():
    driver = FakeDriver()
    async with mcp_session(driver) as session, http_client(driver) as http:
        await session.call_tool("computer_screenshot", {})
        await http.post("/agentenv/ext/computer-display/set", json={"width": 1920, "height": 1080})
        stale = await session.call_tool("computer_click", {"x": 10, "y": 10})
        crop = await session.call_tool("computer_screenshot", {"region": [0, 0, 10, 10]})
        await session.call_tool("computer_screenshot", {})
        fresh = await session.call_tool("computer_click", {"x": 1900, "y": 10})
    assert stale.structuredContent["code"] == "stale_frame"
    assert "captured at 1280×800 at scale 1 and the display is now 1920×1080 at scale 1" in stale.structuredContent["message"]
    assert crop.structuredContent["code"] == "stale_frame" and not fresh.isError


@pytest.mark.asyncio
async def test_with_no_frame_yet_a_point_cannot_be_placed():
    async with mcp_session(FakeDriver()) as session:
        refused = await session.call_tool("computer_click", {"x": 1, "y": 1})
    assert refused.structuredContent["code"] == "stale_frame"
    assert refused.structuredContent["message"] == "there is no frame yet: take a screenshot first"


@pytest.mark.asyncio
async def test_hidpi_frame_pixels_map_to_screen_pixels():
    driver = FakeDriver(FakeScreen(640, 400, scale=2.0))
    async with mcp_session(driver) as session:
        shot = await session.call_tool("computer_screenshot", {})
        await session.call_tool("computer_click", {"x": 21, "y": 41})
        await session.call_tool("computer_click", {"x": 1279, "y": 799})
        position = await session.call_tool("computer_pointer_position", {})
    assert (shot.structuredContent["width"], shot.structuredContent["height"]) == (1280, 800)
    assert shot.structuredContent["summary"] == "Captured f1: 1280×800 of the 640×400 at scale 2 display"
    assert driver.screen.log[-2][1:3] == (10, 20) and driver.screen.log[-1][1:3] == (639, 399)
    assert position.structuredContent["x"] == 1278 and position.structuredContent["screen"] == {"x": 639, "y": 399}


@pytest.mark.asyncio
async def test_each_caller_has_its_own_latest_frame():
    class ByRole(FakeDriver):
        def caller_identity(self, context):
            return context.caller.role or "default"

    driver = ByRole()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {}, meta={ROLE_META_KEY: "alice"})
        await session.call_tool("computer_screenshot", {"region": [0, 0, 10, 10]}, meta={ROLE_META_KEY: "alice"})
        await session.call_tool("computer_screenshot", {}, meta={ROLE_META_KEY: "bob"})
        alice = await session.call_tool("computer_click", {"x": 9, "y": 9}, meta={ROLE_META_KEY: "alice"})
        bob = await session.call_tool("computer_click", {"x": 900, "y": 9}, meta={ROLE_META_KEY: "bob"})
        carol = await session.call_tool("computer_click", {"x": 1, "y": 1}, meta={ROLE_META_KEY: "carol"})
    assert alice.structuredContent["summary"].endswith("in f2") and bob.structuredContent["summary"].endswith("in f3")
    assert carol.structuredContent["code"] == "stale_frame"
