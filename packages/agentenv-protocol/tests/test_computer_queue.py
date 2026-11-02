"""Input: the input tools' parameters and refusals, the queue that runs input one call at a time in arrival order,
others' input since a frame, screenshots after acting, and the pointer and keyboard groups."""
from __future__ import annotations

import asyncio
import time

import anyio
import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import COLOURS, FakeDriver, FakeScreen, MinimalDriver, http_client, mcp_session  # noqa: E402

from agentenv_protocol import ROLE_META_KEY  # noqa: E402
from agentenv_protocol.computer import ComputerSettings, pointer_position  # noqa: E402
from agentenv_protocol.computer.queue import InputQueue  # noqa: E402
from agentenv_protocol.tool_context import EMPTY  # noqa: E402


class ByRole(FakeDriver):
    def caller_identity(self, context):
        return context.caller.role or "default"


async def _shot(session, **meta) -> dict:
    return (await session.call_tool("computer_screenshot", {}, meta=meta or None)).structuredContent


@pytest.mark.asyncio
async def test_a_click_reaches_the_screen_with_its_button_count_and_modifiers():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        result = await session.call_tool("computer_click", {"x": 5, "y": 6, "button": "right", "count": 2,
                                                             "modifiers": ["shift", "Control"]})
    assert result.structuredContent == {
        "summary": "Clicked right ×2 at (5, 6) in f1", "applied": "yes", "delivery": "foreground",
        "others_since_frame": 0, "effect": {"pointer_moved": True}}
    assert driver.screen.log[-1] == ("click", 5, 6, "right", 2, ("Shift", "Control"))
    assert driver.screen.colour_at(5, 6) == COLOURS["right"]


@pytest.mark.asyncio
@pytest.mark.parametrize("args, code, message", [
    ({"button": "back"}, "unsupported", "this computer has no back button; it has left, right, middle"),
    ({"count": 4}, "invalid_argument", "count is from 1 to 3 on this computer, got 4"),
    ({"count": 0}, "invalid_argument", "count is from 1 to 3"),
    ({"modifiers": ["Hyperdrive"]}, "invalid_argument", "'Hyperdrive' is not a W3C key value"),
    ({"x": 5, "y": None}, "invalid_argument", "give x and y, or neither to act at the pointer"),
    ({"x": None, "y": None, "frame_id": "f1"}, "invalid_argument", "give x and y"),
])
async def test_a_bad_click_is_refused_and_nothing_is_sent(args, code, message):
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        refused = await session.call_tool("computer_click", {"x": 5, "y": 6, **args})
    assert refused.structuredContent["code"] == code and message in refused.structuredContent["message"]
    assert refused.structuredContent["applied"] == "no" and driver.screen.log == []


@pytest.mark.asyncio
async def test_without_the_pointer_group_a_click_needs_its_point():
    async with mcp_session(MinimalDriver()) as session:
        await _shot(session)
        refused = await session.call_tool("computer_click", {"x": 5})
    assert refused.isError and "give x and y" in refused.content[0].text


@pytest.mark.asyncio
async def test_a_drag_checks_every_point_before_sending_anything():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        dragged = await session.call_tool("computer_drag", {"path": [[1, 1], [50, 60], [99, 98]], "duration_ms": 300})
        out = await session.call_tool("computer_drag", {"path": [[1, 1], [1280, 5]]})
        short = await session.call_tool("computer_drag", {"path": [[1, 1]]})
        long = await session.call_tool("computer_drag", {"path": [[1, 1]] * 11})
        slow = await session.call_tool("computer_drag", {"path": [[1, 1], [2, 2]], "duration_ms": 10_001})
        bad = await session.call_tool("computer_drag", {"path": [[1, 1], [2, 2, 3]]})
        middle = await session.call_tool("computer_drag", {"path": [[1, 1], [2, 2]], "button": "middle"})
    assert dragged.structuredContent["summary"] == "Dragged left through 3 points of f1 in 300 ms"
    assert driver.screen.log[0] == ("drag", ((1, 1), (50, 60), (99, 98)), "left", 300, ())
    assert out.structuredContent["code"] == "out_of_bounds"
    assert short.structuredContent["code"] == long.structuredContent["code"] == "invalid_argument"
    assert slow.structuredContent["code"] == bad.structuredContent["code"] == "invalid_argument"
    assert not middle.isError and driver.screen.log[1] == ("drag", ((1, 1), (2, 2)), "middle", 500, ())
    assert len(driver.screen.log) == 2


@pytest.mark.asyncio
async def test_scroll_moves_the_content_and_can_scroll_at_the_pointer():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        await session.call_tool("computer_scroll", {"x": 10, "y": 10, "direction": "down"})
        await session.call_tool("computer_pointer_move", {"x": 30, "y": 40})
        at_pointer = await session.call_tool("computer_scroll", {"direction": "left", "amount": 2})
        refused = await session.call_tool("computer_scroll", {"x": 1, "y": 1, "direction": "up", "amount": 101})
    assert driver.screen.scrolled == [-2, 3]
    assert at_pointer.structuredContent["summary"] == "Scrolled left 2 at the pointer, screen (30, 40)"
    assert driver.screen.log[-1] == ("scroll", 30, 40, "left", 2, ())
    assert refused.structuredContent["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_a_click_without_a_point_clicks_at_the_pointer():
    driver = FakeDriver()
    driver.screen.pointer = (70, 80)
    async with mcp_session(driver) as session:
        clicked = await session.call_tool("computer_click", {})
    assert clicked.structuredContent["others_since_frame"] == 0 and driver.screen.log[-1][1:3] == (70, 80)


@pytest.mark.asyncio
@pytest.mark.parametrize("text, code", [("", "invalid_argument"), ("a" * 10_001, "invalid_argument"),
                                        ("bell\x07", "invalid_argument"), ("del\x7f", "invalid_argument")])
async def test_text_that_cant_be_typed_whole_is_refused(text, code):
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        refused = await session.call_tool("computer_type", {"text": text})
    assert refused.structuredContent["code"] == code and driver.screen.log == []


@pytest.mark.asyncio
async def test_unicode_newlines_and_tabs_are_typed():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        typed = await session.call_tool("computer_type", {"text": "Grüße 東京\n\tok", "screenshot": True})
    assert driver.screen.log == [("type", "Grüße 東京\n\tok")]
    assert typed.structuredContent["summary"] == "Typed 12 characters, then captured f1"
    assert typed.structuredContent["frame"]["frame_id"] == "f1" and typed.content[1].type == "image"


@pytest.mark.asyncio
async def test_a_key_chord_is_pressed_in_w3c_case_and_repeated_by_the_base_class():
    driver = FakeDriver(settings=ComputerSettings(tap_hold_ms=25))
    async with mcp_session(driver) as session:
        pressed = await session.call_tool("computer_key", {"keys": ["control", "S"], "repeat": 3})
        single = await session.call_tool("computer_key", {"keys": [" "]})
    assert pressed.structuredContent["summary"] == "Pressed Control+S ×3"
    assert driver.screen.log == [("press", ("Control", "S"), 25)] * 3 + [("press", (" ",), 25)]
    assert single.structuredContent["applied"] == "yes"


@pytest.mark.asyncio
@pytest.mark.parametrize("keys, repeat, code, hint", [
    (["ctrl", "s"], 1, "invalid_argument", "did you mean Control?"),
    (["Control+s"], 1, "invalid_argument", "one key per entry: keys is a list, such as ['Control', 's']"),
    (["Entr"], 1, "invalid_argument", "did you mean Enter?"),
    (["Qwertyuiop"], 1, "invalid_argument", "named keys are W3C UI Events key values, such as Enter"),
    ([], 1, "invalid_argument", None),
    (["Enter"], 101, "invalid_argument", None),
])
async def test_a_refused_key_says_what_was_meant(keys, repeat, code, hint):
    async with mcp_session(FakeDriver()) as session:
        refused = await session.call_tool("computer_key", {"keys": keys, "repeat": repeat})
    assert refused.structuredContent["code"] == code and refused.structuredContent.get("hint") == hint


@pytest.mark.asyncio
async def test_a_key_that_isnt_a_string_is_refused_by_the_tools_schema():
    async with mcp_session(FakeDriver()) as session:
        refused = await session.call_tool("computer_key", {"keys": [3]})
    assert refused.isError and refused.structuredContent is None


@pytest.mark.asyncio
async def test_a_driver_that_declares_its_keys_refuses_the_others():
    class Few(FakeDriver):
        keys = ("Enter", "a", "Control")

    driver = Few()
    async with mcp_session(driver) as session:
        refused = await session.call_tool("computer_key", {"keys": ["Control", "Tab"]})
        allowed = await session.call_tool("computer_key", {"keys": ["control", "a"]})
    assert refused.structuredContent["code"] == "unsupported" and "['Tab']" in refused.structuredContent["message"]
    assert not allowed.isError


@pytest.mark.asyncio
async def test_wait_waits_within_range_and_can_return_a_frame():
    async with mcp_session(FakeDriver()) as session:
        started = time.monotonic()
        waited = await session.call_tool("computer_wait", {"ms": 120})
        elapsed = time.monotonic() - started
        shot = await session.call_tool("computer_wait", {"ms": 0, "screenshot": True})
        refused = await session.call_tool("computer_wait", {"ms": 60_001})
    assert waited.structuredContent == {"summary": "Waited 120 ms", "waited_ms": 120}
    assert elapsed >= 0.1
    assert shot.structuredContent["frame"]["frame_id"] == "f1" and shot.content[1].type == "image"
    assert refused.structuredContent["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_input_runs_one_call_at_a_time_in_arrival_order():
    driver = FakeDriver()
    driver.slow["click"] = 0.05
    async with mcp_session(driver) as session:
        await _shot(session)
        results = await asyncio.gather(*(session.call_tool("computer_click", {"x": i, "y": i}) for i in range(6)))
    assert all(not r.isError for r in results)
    assert [entry[1] for entry in driver.screen.log] == list(range(6))


@pytest.mark.asyncio
async def test_a_call_waiting_past_the_limit_is_refused_input_busy():
    driver = FakeDriver(settings=ComputerSettings(queue_wait_s=0.1))
    driver.slow["click"] = 0.4
    async with mcp_session(driver) as session:
        await _shot(session)
        first, second = await asyncio.gather(session.call_tool("computer_click", {"x": 1, "y": 1}),
                                             session.call_tool("computer_key", {"keys": ["a"]}))
    assert not first.isError
    assert second.structuredContent == {"code": "input_busy", "message": "waited 0.1 s for earlier input to finish",
                                        "retryable": True, "applied": "no"}


@pytest.mark.asyncio
async def test_screenshots_and_waits_never_queue_behind_input():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        driver.slow["click"] = 0.5
        click = asyncio.ensure_future(session.call_tool("computer_click", {"x": 1, "y": 1}))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await session.call_tool("computer_screenshot", {})
        await session.call_tool("computer_wait", {"ms": 0})
        observed = time.monotonic() - started
        await click
    assert observed < 0.4


@pytest.mark.asyncio
async def test_others_input_since_a_frame_is_counted_and_ones_own_is_not():
    driver = ByRole()
    async with mcp_session(driver) as session:
        alice_frame = await _shot(session, **{ROLE_META_KEY: "alice"})
        await session.call_tool("computer_click", {"x": 1, "y": 1}, meta={ROLE_META_KEY: "alice"})
        await _shot(session, **{ROLE_META_KEY: "bob"})
        for _ in range(2):
            await session.call_tool("computer_key", {"keys": ["a"]}, meta={ROLE_META_KEY: "bob"})
        alice = await session.call_tool("computer_click", {"x": 2, "y": 2, "frame_id": alice_frame["frame_id"]},
                                        meta={ROLE_META_KEY: "alice"})
        bob = await session.call_tool("computer_key", {"keys": ["b"]}, meta={ROLE_META_KEY: "bob"})
    assert alice.structuredContent["others_since_frame"] == 2
    assert bob.structuredContent["others_since_frame"] == 1


@pytest.mark.asyncio
async def test_an_input_can_return_a_fresh_frame_that_shows_it():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        clicked = await session.call_tool("computer_click", {"x": 3, "y": 3, "screenshot": True})
    assert clicked.structuredContent["frame"]["frame_id"] == "f2"
    assert clicked.structuredContent["summary"] == "Clicked left at (3, 3) in f1, then captured f2"


@pytest.mark.asyncio
async def test_the_pointer_tools():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        before = await session.call_tool("computer_pointer_position", {})
        await _shot(session)
        moved = await session.call_tool("computer_pointer_move", {"x": 11, "y": 12})
        down = await session.call_tool("computer_pointer_down", {"button": "right", "x": 20, "y": 21})
        up = await session.call_tool("computer_pointer_up", {"button": "right"})
        inside = await session.call_tool("computer_pointer_position", {})
        await session.call_tool("computer_screenshot", {"region": [0, 0, 10, 10]})
        outside = await session.call_tool("computer_pointer_position", {})
        named = await session.call_tool("computer_pointer_position", {"frame_id": "f1"})
        half = await session.call_tool("computer_pointer_down", {"x": 1})
        listed = await session.call_tool("computer_pointer_list", {})
    assert before.structuredContent == {"summary": "The pointer is at screen (0, 0)", "screen": {"x": 0, "y": 0}}
    assert moved.structuredContent["summary"] == "Moved the pointer at (11, 12) in f1"
    assert down.structuredContent["summary"] == "Pressed right at (20, 21) in f1"
    assert up.structuredContent["summary"] == "Released right"
    assert driver.screen.log == [("move", 11, 12), ("move", 20, 21), ("down", "right"), ("up", "right")]
    assert inside.structuredContent["inside_frame"] is True and inside.structuredContent["x"] == 20
    assert outside.structuredContent["inside_frame"] is False and "outside f2" in outside.structuredContent["summary"]
    assert named.structuredContent["frame_id"] == "f1"
    assert half.structuredContent["code"] == "invalid_argument"
    assert listed.structuredContent["pointers"] == [
        {"id": "default", "kind": "foreground", "label": "The computer's pointer", "default": True}]


@pytest.mark.asyncio
async def test_a_bad_pointer_position_is_an_internal_error():
    class Lost(FakeDriver):
        @pointer_position
        async def where(self):
            return "somewhere"

    async with mcp_session(Lost()) as session:
        broken = await session.call_tool("computer_pointer_position", {})
    assert broken.structuredContent["code"] == "internal"


@pytest.mark.asyncio
async def test_keys_held_down_stay_down_until_released():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_keyboard_down", {"keys": ["shift"]})
        await session.call_tool("computer_keyboard_down", {"keys": ["Control"]})
        held = list(driver.screen.held)
        partly = await session.call_tool("computer_keyboard_up", {"keys": ["Shift"]})
        rest = await session.call_tool("computer_keyboard_up", {})
        nothing = await session.call_tool("computer_keyboard_up", {})
    assert held == ["Shift", "Control"] and driver.screen.held == []
    assert partly.structuredContent["summary"] == "Released Shift" and rest.structuredContent["summary"] == "Released Control"
    assert nothing.structuredContent["code"] == "invalid_argument" and driver._computer.held == {}


@pytest.mark.asyncio
async def test_a_hold_with_hold_ms_releases_in_reverse_order():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        held = await session.call_tool("computer_keyboard_down", {"keys": ["Shift", "a"], "hold_ms": 30})
        refused = await session.call_tool("computer_keyboard_down", {"keys": ["a"], "hold_ms": 10_001})
    assert held.structuredContent["summary"] == "Held Shift+a for 30 ms"
    assert driver.screen.log == [("key_down", ("Shift", "a")), ("key_up", ("a", "Shift"))]
    assert refused.structuredContent["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_keys_held_past_max_hold_ms_are_released():
    driver = FakeDriver(settings=ComputerSettings(max_hold_ms=50))
    async with mcp_session(driver) as session:
        await session.call_tool("computer_keyboard_down", {"keys": ["Alt"]})
        with anyio.fail_after(5):
            while driver.screen.held:
                await asyncio.sleep(0.01)
    assert driver.screen.log[-1] == ("key_up", ("Alt",)) and driver._computer.held == {}


@pytest.mark.asyncio
async def test_a_second_hold_restarts_the_release_timer():
    """Timed by the clock: the keys go up no sooner than max_hold_ms after the second hold was sent, where the first
    hold's timer would have released them 0.3 s sooner, and a slow loop has 1.2 s to land the second hold before it."""
    driver = FakeDriver(settings=ComputerSettings(max_hold_ms=1500))
    async with mcp_session(driver) as session:
        await session.call_tool("computer_keyboard_down", {"keys": ["Alt"]})
        await asyncio.sleep(0.3)
        second = time.monotonic()
        await session.call_tool("computer_keyboard_down", {"keys": ["Shift"]})
        both = list(driver.screen.held)
        with anyio.fail_after(10):
            while driver.screen.held:
                await asyncio.sleep(0.01)
        released = time.monotonic()
    assert both == ["Alt", "Shift"] and driver.screen.log[-1] == ("key_up", ("Shift", "Alt"))
    assert released - second >= 1.5


@pytest.mark.asyncio
async def test_the_minimal_driver_serves_no_group_tools():
    driver = MinimalDriver(FakeScreen(64, 40))
    async with mcp_session(driver) as session:
        await _shot(session)
        clicked = await session.call_tool("computer_click", {"x": 3, "y": 4})
        missing = await session.call_tool("computer_pointer_move", {"x": 1, "y": 1})
    assert not clicked.isError and driver.screen.log == [("click", 3, 4, "left", 1, ())]
    assert missing.isError and "Unknown tool" in missing.content[0].text


@pytest.mark.asyncio
async def test_a_caller_cancelled_while_waiting_never_leaves_the_queue_held():
    queue = InputQueue()

    async def hold():
        async with queue.turn(5):
            await asyncio.sleep(0.2)

    async def wait():
        async with queue.turn(5):
            pass

    holder = asyncio.ensure_future(hold())
    await asyncio.sleep(0.01)
    waiter = asyncio.ensure_future(wait())
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await holder
    async with queue.turn(0.5):
        pass


@pytest.mark.asyncio
async def test_a_screenshot_that_fails_after_the_input_reports_the_input_as_done():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await _shot(session)
        driver.fail["capture"] = RuntimeError("mss died")
        clicked = await session.call_tool("computer_click", {"x": 4, "y": 4, "screenshot": True})
        driver.fail["capture"] = RuntimeError("mss died")
        held = await session.call_tool("computer_keyboard_down", {"keys": ["Shift"], "screenshot": True})
    assert not clicked.isError and clicked.structuredContent["applied"] == "yes" and "frame" not in clicked.structuredContent
    assert "the screenshot after it failed (internal: @capture failed: RuntimeError: mss died)" in clicked.content[0].text
    assert [e[0] for e in driver.screen.log] == ["click", "key_down"]
    assert not held.isError and driver._computer.held == {"default": ["Shift"]}


@pytest.mark.asyncio
async def test_the_automatic_release_waits_out_a_busy_queue():
    driver = FakeDriver(settings=ComputerSettings(max_hold_ms=50, queue_wait_s=0.05))
    async with mcp_session(driver) as session:
        await _shot(session)
        await session.call_tool("computer_keyboard_down", {"keys": ["Alt"]})
        driver.slow["drag"] = 0.4
        await session.call_tool("computer_drag", {"path": [[1, 1], [9, 9]]})
        with anyio.fail_after(5):
            while driver.screen.held:
                await asyncio.sleep(0.01)
    kinds = [e[0] for e in driver.screen.log]
    assert kinds == ["key_down", "drag", "key_up"] and driver._computer.held == {}


@pytest.mark.asyncio
async def test_a_point_queued_while_the_display_changes_is_refused_stale():
    driver = FakeDriver()
    async with mcp_session(driver) as session, http_client(driver) as http:
        await _shot(session)
        driver.slow["click"] = 0.3
        first = asyncio.ensure_future(session.call_tool("computer_click", {"x": 1, "y": 1}))
        await asyncio.sleep(0.05)
        resize = asyncio.ensure_future(http.post("/agentenv/ext/computer-display/set", json={"width": 640, "height": 400}))
        await asyncio.sleep(0.05)
        queued = await session.call_tool("computer_click", {"x": 1270, "y": 10})
        await first
        await resize
    assert queued.structuredContent["code"] == "stale_frame" and queued.structuredContent["applied"] == "no"
    assert [e[1:3] for e in driver.screen.log] == [(1, 1)]


@pytest.mark.asyncio
async def test_a_click_at_the_pointer_reads_the_pointer_when_its_turn_comes():
    driver = FakeDriver()
    driver.slow["drag"] = 0.3
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        await session.call_tool("computer_pointer_move", {"x": 5, "y": 5})
        dragging = asyncio.ensure_future(session.call_tool("computer_drag", {"path": [[10, 10], [100, 100]]}))
        await asyncio.sleep(0.1)
        clicked = await session.call_tool("computer_click", {})
        await dragging
    assert clicked.structuredContent["summary"] == "Clicked left at the pointer, screen (100, 100)"
    assert driver.screen.log[-1][:3] == ("click", 100, 100)


@pytest.mark.asyncio
async def test_input_that_lands_while_a_screenshot_is_taken_counts_as_after_it():
    driver = ByRole()
    async with mcp_session(driver) as session:
        await _shot(session, **{ROLE_META_KEY: "bob"})
        driver.slow["capture"] = 0.3
        shot = asyncio.ensure_future(_shot(session, **{ROLE_META_KEY: "alice"}))
        await asyncio.sleep(0.1)
        await session.call_tool("computer_click", {"x": 1, "y": 1}, meta={ROLE_META_KEY: "bob"})
        frame = await shot
        del driver.slow["capture"]
        clicked = await session.call_tool("computer_click", {"x": 2, "y": 2, "frame_id": frame["frame_id"]},
                                          meta={ROLE_META_KEY: "alice"})
    assert clicked.structuredContent["others_since_frame"] == 1


@pytest.mark.asyncio
async def test_a_timed_hold_that_is_cancelled_still_lets_go_of_its_keys():
    driver = FakeDriver()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
    holding = asyncio.ensure_future(driver.computer_keyboard_down(EMPTY, keys=["Shift"], hold_ms=5000))
    await asyncio.sleep(0.1)
    holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding
    assert driver.screen.log[-2:] == [("key_down", ("Shift",)), ("key_up", ("Shift",))]
