"""The conformance fixture, faked on the fake driver: the Fixture Desk's layout, its markers painted on the fake screen,
and its state built from the input that reaches the screen, all served through the driver's data/add and
computer-exec as the real fixture is. Commands that aren't the fixture's run for real, through the Linux default.
``served`` puts a driver on loopback, so the probe uses its own HTTP and MCP clients."""
from __future__ import annotations

import contextlib
import json
import math
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import uvicorn
from computer_fake import FakeDriver

from agentenv_protocol.computer import (
    ComputerDriverEnvironment,
    ExecResult,
    click,
    drag,
    pointer_up,
    press,
    run_in_session,
    scroll,
    type_text,
)
from agentenv_protocol.computer import linux
from agentenv_protocol.computer.conformance.checks import STAGE

CLIPBOARD = "agentenv-probe clipboard"
FIELD = (1080, 500)
MARKERS = ((11, 11, "#ff0000"), (12, 11, "#00ff00"), (11, 12, "#0000ff"), (12, 12, "#ffff00"), (37, 20, "#ff00ff"),
           (20, 37, "#00ffff"))
LAYOUT = {"name": [140, 40, 420, 34], "save": [560, 34, 120, 44], "subscribe": [40, 100, 160, 34],
          "notes": [760, 40, 380, 90], "list": [40, 160, 280, 400], "grid": [400, 160, 400, 400],
          "pad": [840, 160, 400, 300], "wheel": [840, 500, 200, 120], "field": [*FIELD, 48, 48], "list_row_height": 24,
          "markers": [[FIELD[0] + x, FIELD[1] + y, c] for x, y, c in MARKERS], "cell": 50, "clipboard": CLIPBOARD}
KEYSYMS = {" ": "space", ",": "comma", "!": "exclam", "\t": "Tab", "\n": "Return"}


class FakeDesk:
    """What the Fixture Desk would record of the input the fake screen gets."""

    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.started = False
        self.focus: str | None = None
        self.selected_all = False
        self.state = {"name": "", "notes": "", "keys": [], "saved_count": 0, "subscribed": False, "selected": None,
                      "scroll_top": 0.0, "cells": {}, "right_clicks": 0, "double_clicks": 0, "drags": [],
                      "wheel": {"up": 0, "down": 0, "left": 0, "right": 0}, "clipboard": CLIPBOARD, "loaded": None,
                      "screen": [1280, 800]}

    def start(self) -> None:
        screen = self.driver.screen
        for x in range(48):
            for y in range(48):
                screen.paint(FIELD[0] + x, FIELD[1] + y, (0, 0, 0))
        for x, y, colour in MARKERS:
            screen.paint(FIELD[0] + x, FIELD[1] + y, (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)))
        self.started = True

    def read(self) -> dict:
        message = Path.home() / STAGE / "state" / "inbox.txt"
        self.state["loaded"] = message.read_text() if message.exists() else self.state["loaded"]
        return self.state

    def _in(self, box: str, x: float, y: float) -> tuple[float, float] | None:
        x0, y0, w, h = LAYOUT[box]
        return (x - x0, y - y0) if x0 <= x < x0 + w and y0 <= y < y0 + h else None

    def click(self, x: float, y: float, button: str, count: int) -> None:
        if not self.started:
            return
        if (at := self._in("grid", x, y)) is not None:
            key = f"{int(at[1] // 50)},{int(at[0] // 50)}"
            self.state["cells"][key] = (self.state["cells"].get(key, 0) + count) % 4
            self.state["right_clicks"] += button == "right"
            self.state["double_clicks"] += count >= 2
        elif self._in("save", x, y) is not None:
            self.state["saved_count"] += 1
        elif self._in("subscribe", x, y) is not None:
            self.state["subscribed"] = not self.state["subscribed"]
        for box in ("name", "notes"):
            if self._in(box, x, y) is not None:
                self.focus = box

    def type(self, text: str) -> None:
        if self.focus is None:
            return
        held = "Shift" in self.driver.screen.held
        text = text.upper() if held else text
        self.state[self.focus] = ("" if self.selected_all else self.state[self.focus]) + text
        self.selected_all = False
        self.state["keys"] = (self.state["keys"] + [KEYSYMS.get(c, c) for c in text])[-100:]

    def press(self, keys: list[str]) -> None:
        if self.focus is None:
            return
        if keys == ["Control", "/"]:
            self.selected_all = True
        elif keys == ["Backspace"]:
            value = self.state[self.focus]
            self.state[self.focus] = "" if self.selected_all else value[:-1]
            self.selected_all = False

    def scroll(self, x: float, y: float, direction: str, amount: int) -> None:
        if self._in("list", x, y) is not None and direction == "down":
            self.state["scroll_top"] = min(1.0, self.state["scroll_top"] + 0.01 * amount)
        elif self._in("wheel", x, y) is not None:
            self.state["wheel"][direction] += amount

    def drag(self, path: list) -> None:
        local = [self._in("pad", x, y) for x, y in path]
        if None in local:
            return
        stroke = [local[0]]
        for (x0, y0), (x1, y1) in zip(local, local[1:]):
            steps = max(2, int(math.dist((x0, y0), (x1, y1)) // 10))
            stroke += [(x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps) for k in range(1, steps + 1)]
        self.state["drags"].append({"start": list(stroke[0]), "end": list(stroke[-1]), "points": len(stroke),
                                    "path": [list(p) for p in stroke]})

    def release(self) -> None:
        if self.started and self._in("save", *self.driver.screen.pointer) is not None:
            self.state["saved_count"] += 1


class FixtureDriver(FakeDriver):
    """The fake driver with the fake desk behind its input, its data/add and its computer-exec."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.desk = FakeDesk(self)
        self.commands: list[list[str]] = []

    @click
    async def click_at(self, x, y, button, count, modifiers):
        effect = await super().click_at(x, y, button, count, modifiers)
        self.desk.click(x, y, button, count)
        return effect

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers):
        await super().drag_along(path, button, duration_ms, modifiers)
        self.desk.drag(path)

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers):
        await super().scroll_at(x, y, direction, amount, modifiers)
        self.desk.scroll(x, y, direction, amount)

    @type_text
    async def type_in(self, text):
        await super().type_in(text)
        self.desk.type(text)

    @press
    async def press_keys(self, keys):
        await super().press_keys(keys)
        self.desk.press(list(keys))

    @pointer_up
    async def up(self, button):
        await super().up(button)
        self.desk.release()

    @run_in_session
    async def run(self, argv, cwd, env, timeout_s):
        self.commands.append(list(argv))
        answer = self._fixture(argv)
        if answer is not None:
            return answer
        return await linux.run_in_session(argv, cwd, env, timeout_s, output_bytes=self.settings.exec_output_bytes)

    def _fixture(self, argv: list[str]) -> ExecResult | None:
        text = " ".join(argv)
        if argv[:1] == ["cat"] and argv[1].endswith("layout.json"):
            return ExecResult(0, json.dumps(LAYOUT).encode()) if self.desk.started else ExecResult(1, b"", b"missing")
        if argv[:1] == ["cat"] and argv[1].endswith("state.json"):
            return ExecResult(0, json.dumps(self.desk.read()).encode()) if self.desk.started else ExecResult(1)
        if f"python3 {STAGE}/fixture.py" in text:
            self.desk.start()
            return ExecResult(0, b"4242\n")
        if argv in (["id", "-u"], ["stat", "-c", "%u", f"{STAGE}/fixture.py"]):
            return ExecResult(0, b"1000\n")
        if argv[:1] in (["kill"], ["tail"]) or f"rm -rf {STAGE}/state" in text:
            return ExecResult(0)
        return None


@contextlib.contextmanager
def served(driver: ComputerDriverEnvironment) -> Iterator[str]:
    """``driver`` on loopback in a thread of its own: the URL its card is served at."""
    app = (driver.mcp or driver.create_app()).streamable_http_app()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(5)
