"""MacOSDriver, the computer/v1 operations on macOS, against a fake Mac that records the Quartz events it's sent: keys
pressed as a US keyboard sends them, modifiers around what they modify, held keys and buttons on every later event,
drags through their waypoints, captures as BGRA, and the TCC permissions computer-ready reports."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from agentenv_protocol.computer import (  # noqa: E402
    ComputerSettings,
    DeliveryFailed,
    Image,
    MacOSDriver,
    Platform,
    Unsupported,
    computer_driver,
)
from agentenv_protocol.computer import _macos  # noqa: E402

MACOS = Platform(os="macos", arch="arm64", display="aqua")
SHIFT, CONTROL, COMMAND = 0x20000, 0x40000, 0x100000
LEFT_DOWN, LEFT_UP, LEFT_DRAGGED, MOVED = 1, 2, 6, 5


class FakeMac:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.at = (0.0, 0.0)
        self.lacking: list[str] = []
        self.fails = False

    def missing(self) -> list[str]:
        return self.lacking

    def display(self) -> tuple[int, int, float]:
        return 1920, 1080, 1.0

    def capture(self) -> tuple[int, int, bytes]:
        return 2, 1, bytes([1, 2, 3, 255, 4, 5, 6, 255])

    def mouse(self, kind, x, y, button, count, flags) -> None:
        if self.fails:
            raise ValueError("no event")
        self.sent.append(("mouse", kind, x, y, button, count, flags))
        self.at = (x, y)

    def wheel(self, x, y, lines, across, flags) -> None:
        self.sent.append(("wheel", x, y, lines, across, flags))

    def key(self, code, down, flags, character=None) -> None:
        self.sent.append(("key", code, down, flags, character))

    def location(self) -> tuple[float, float]:
        return self.at


@computer_driver(name="desk", supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}])
class Desk(MacOSDriver):
    pass


@pytest.fixture
def mac(monkeypatch) -> FakeMac:
    fake = FakeMac()
    monkeypatch.setattr(_macos, "Quartz", lambda: fake)
    monkeypatch.setattr(_macos, "SETTLE_S", 0)
    monkeypatch.setattr(_macos, "CLICK_GAP_S", 0)
    monkeypatch.setattr(_macos, "CLICK_HOLD_S", 0)
    monkeypatch.setattr(_macos, "SCROLL_GAP_S", 0)
    return fake


@pytest.fixture
def desk(mac) -> Desk:
    return Desk(ComputerSettings(tap_hold_ms=0, type_delay_ms=0), MACOS)


def keys(sent: list[tuple]) -> list[tuple]:
    return [event[1:] for event in sent if event[0] == "key"]


@pytest.mark.asyncio
async def test_typed_text_goes_out_as_the_keys_of_a_us_keyboard_and_text_off_it_as_the_characters(desk, mac):
    await desk.type_in("aB ,\n\tö")

    assert keys(mac.sent) == [
        (0, True, 0, None), (0, False, 0, None),
        (56, True, SHIFT, None), (11, True, SHIFT, None), (11, False, SHIFT, None), (56, False, 0, None),
        (49, True, 0, None), (49, False, 0, None),
        (43, True, 0, None), (43, False, 0, None),
        (36, True, 0, None), (36, False, 0, None),
        (48, True, 0, None), (48, False, 0, None),
        (0, True, 0, "ö"), (0, False, 0, "ö"),
    ]


@pytest.mark.asyncio
async def test_a_held_key_stays_down_on_every_event_until_released(desk, mac):
    await desk.keys_down(["Shift"])
    await desk.type_in("a")
    await desk.click_at(10, 20, "left", 1, [])
    await desk.keys_up(["Shift"])
    await desk.type_in("d")

    assert keys(mac.sent)[:3] == [(56, True, SHIFT, None), (0, True, SHIFT, None), (0, False, SHIFT, None)]
    assert [event[-1] for event in mac.sent if event[0] == "mouse"] == [SHIFT, SHIFT, SHIFT]
    assert keys(mac.sent)[3:] == [(56, False, 0, None), (2, True, 0, None), (2, False, 0, None)]


@pytest.mark.asyncio
async def test_a_chord_presses_its_modifiers_around_its_key_in_order_and_releases_them_in_reverse_shift_too(desk,
                                                                                                            mac):
    await desk.press_keys(["Control", "Meta", "/"])
    await desk.press_keys(["Shift"])

    assert keys(mac.sent) == [
        (59, True, CONTROL, None), (55, True, CONTROL | COMMAND, None),
        (44, True, CONTROL | COMMAND, None), (44, False, CONTROL | COMMAND, None),
        (55, False, CONTROL, None), (59, False, 0, None),
        (56, True, SHIFT, None), (56, False, 0, None),
    ]
    with pytest.raises(Unsupported, match="no key on a US keyboard"):
        await desk.press_keys(["é"])
    mac.sent.clear()
    await desk.press_keys(["Control", "A"])
    await desk.press_keys(["!"])

    assert keys(mac.sent) == [
        (59, True, CONTROL, None), (56, True, CONTROL | SHIFT, None), (0, True, CONTROL | SHIFT, None),
        (0, False, CONTROL | SHIFT, None), (56, False, CONTROL, None), (59, False, 0, None),
        (56, True, SHIFT, None), (18, True, SHIFT, None), (18, False, SHIFT, None), (56, False, 0, None),
    ]


@pytest.mark.asyncio
async def test_a_click_moves_there_first_and_counts_its_presses(desk, mac):
    await desk.click_at(100, 50, "left", 2, ["Meta"])

    mouse = [event[1:] for event in mac.sent if event[0] == "mouse"]
    assert mouse == [(MOVED, 100, 50, 0, 1, COMMAND), (LEFT_DOWN, 100, 50, 0, 1, COMMAND),
                     (LEFT_UP, 100, 50, 0, 1, COMMAND), (LEFT_DOWN, 100, 50, 0, 2, COMMAND),
                     (LEFT_UP, 100, 50, 0, 2, COMMAND)]
    assert keys(mac.sent) == [(55, True, COMMAND, None), (55, False, 0, None)]


@pytest.mark.asyncio
async def test_a_drag_presses_at_its_start_moves_through_its_waypoints_and_releases_at_its_end(desk, mac):
    await desk.drag_along([(0, 0), (40, 0), (40, 30)], "left", 0, [])

    kinds = [(event[1], event[2], event[3]) for event in mac.sent]
    assert kinds[:2] == [(MOVED, 0, 0), (LEFT_DOWN, 0, 0)] and kinds[-1] == (LEFT_UP, 40, 30)
    moved = [(x, y) for kind, x, y in kinds if kind == LEFT_DRAGGED]
    assert (40, 0) in moved and moved[-1] == (40, 30)
    assert all(abs(x1 - x0) + abs(y1 - y0) <= _macos.DRAG_STEP_PX for (x0, y0), (x1, y1) in zip(moved, moved[1:]))


@pytest.mark.asyncio
async def test_the_pointer_group_holds_a_button_across_moves(desk, mac):
    await desk.move(5, 5)
    await desk.down("left")
    await desk.move(30, 40)
    await desk.up("left")
    await desk.move(31, 41)

    assert [event[1:4] for event in mac.sent] == [(MOVED, 5, 5), (LEFT_DOWN, 5, 5), (LEFT_DRAGGED, 30, 40),
                                                  (LEFT_UP, 30, 40), (MOVED, 31, 41)]
    assert await desk.where() == (31, 41)


@pytest.mark.asyncio
async def test_a_scroll_sends_one_wheel_line_per_step_where_it_points(desk, mac):
    await desk.scroll_at(7, 8, "down", 3, [])
    await desk.scroll_at(7, 8, "left", 1, [])

    assert [event for event in mac.sent if event[0] == "wheel"] == [("wheel", 7, 8, -1, 0, 0)] * 3 + [
        ("wheel", 7, 8, 0, 1, 0)]


@pytest.mark.asyncio
async def test_a_capture_is_bgra_and_the_display_its_size(desk):
    shot = await desk.grab()
    screen = await desk.screen()

    assert shot == Image(2, 1, bytes([1, 2, 3, 255, 4, 5, 6, 255]), "BGRA")
    assert (screen.width, screen.height, screen.scale) == (1920, 1080, 1.0)


@pytest.mark.asyncio
async def test_computer_ready_names_the_permissions_macos_hasnt_granted(desk, mac):
    (granted,) = await desk.ready_checks()
    mac.lacking = ["Accessibility", "Screen Recording"]
    (lacking,) = await desk.ready_checks()

    assert granted.ok
    assert not lacking.ok and "Accessibility, Screen Recording" in lacking.detail


@pytest.mark.asyncio
async def test_input_quartz_fails_to_send_is_a_failed_delivery(desk, mac):
    mac.fails = True
    with pytest.raises(DeliveryFailed, match="Quartz failed"):
        await desk.click_at(1, 1, "left", 1, [])


def test_its_card_declares_the_pointer_and_keyboard_groups_and_the_macos_defaults():
    card = Desk.declaration(MACOS)
    tools = {tool.name for tool in card.capabilities.tools}
    uris = {extension.uri for extension in card.capabilities.extensions}

    assert {"computer_pointer_move", "computer_keyboard_down", "computer_keyboard_up"} <= tools
    assert {"urn:agentenv:computer-pointer/v1", "urn:agentenv:computer-keyboard/v1",
            "urn:agentenv:computer-exec/v1"} <= uris
