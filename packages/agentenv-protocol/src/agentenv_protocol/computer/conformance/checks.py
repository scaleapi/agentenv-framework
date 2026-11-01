"""The checks a probe runs, named as the driver design names them: P0 to P9 against the contract, and X for one
extension's own methods. Everything goes through the driver's card, tools and extensions; ``full`` also stages the
fixture with data/add, launches it with computer-exec and reads its state back the same way."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from importlib import resources

from mcp import ClientSession
from mcp.types import CallToolResult
from pydantic import ValidationError

from ... import client as protocol
from ...client import ExtensionCallError
from ...tool_access import DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI
from ...types import DATA_OBJECTS_EXTENSION_URI, INTAKE_EXTENSION_URI, SKILL_EXTENSION_URI
from ..card import (
    COMPUTER,
    COMPUTER_DISPLAY,
    COMPUTER_EXEC,
    COMPUTER_KEYBOARD,
    COMPUTER_POINTER,
    COMPUTER_READY,
    COMPUTER_RECORDING,
    COMPUTER_SPACES,
    GROUP_TOOLS,
    POINT_TOOLS,
)
from ..manifest import RecordingManifest
from ..recording import RECORDINGS_DIR
from .client import Driver, image
from .mp4 import read_mp4
from .report import CheckResult

MINIMUM_TOOLS = ("computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_type",
                 "computer_key", "computer_wait")
DATA_PLANE = (INTAKE_EXTENSION_URI, DATA_OBJECTS_EXTENSION_URI)
#: Where the fixture is staged and keeps its state, relative to the session user's home.
STAGE = ".agentenv-probe"
#: What the tool-access check's role, new each run, and the probe's request ids start with.
PROBE_ROLE = "agentenv-probe"
#: How long the probe waits for an effect to reach the fixture, and how often it looks.
SETTLE_S = 5.0
POLL_S = 0.15
#: How long the fixture may take to start.
FIXTURE_START_S = 30.0
NORM = 1000
#: How long the probe records, and the frame rate it asks for.
RECORD_S = 2.0
RECORD_FPS = 5
#: Prints a file's size and SHA-256 as JSON, run with ``python3 -c`` on the computer.
READ_DIGEST = ("import hashlib,json,sys;d=open(sys.argv[1],'rb').read();"
               "print(json.dumps({'size':len(d),'sha256':hashlib.sha256(d).hexdigest()}))")
#: Prints ``length`` bytes of a file from ``offset``, base64, under computer-exec's output cap.
READ_CHUNK = ("import base64,sys;f=open(sys.argv[1],'rb');f.seek(int(sys.argv[2]));"
              "print(base64.b64encode(f.read(int(sys.argv[3]))).decode())")
CHUNK_BYTES = 512 << 10
#: The checks only ``full`` runs, recorded unverified when the fixture can't be staged.
FULL_CHECKS = (("P3", "fixture colours land where the mapping says"), ("P4", "clicks show their effect"),
               ("P5", "refusals leave the fixture as it was"), ("P6", "typed and pressed keys arrive"),
               ("P7", "scrolls and drags arrive"), ("P9", "commands and data/add run as one unprivileged user"))

Test = Callable[[], Awaitable[str | None]]


class Probe:
    """One probe run: the driver, an MCP session on its tools, and the results so far. A test returns what went
    wrong, or None when its check passes; a test that raises fails with what it raised."""

    def __init__(self, driver: Driver, session: ClientSession, ready_timeout_s: float) -> None:
        self.driver = driver
        self.session = session
        self.ready_timeout_s = ready_timeout_s
        self.results: list[CheckResult] = []
        self.features = driver.params(COMPUTER)
        self.layout: dict = {}
        self.fixture_pid: str | None = None

    async def basic(self) -> None:
        for check, section in (("P0", self._ready), ("P1", self._card), ("P2", self._frame), ("P5", self._refusals),
                               ("X", self._display), ("X", self._spaces), ("X", self._keyboard_settings),
                               ("X", self._exec), ("X", self._data_get), ("X", self._app_exports),
                               ("X", self._tool_access), ("X", self._pointer), ("X", self._skill)):
            await self._guarded(check, section)
        self._grants()

    async def full(self) -> None:
        missing = [uri for uri in (COMPUTER_EXEC,) if not self.driver.declares(uri)]
        if not (protocol.intake_declaration(self.driver.card) or {}).get("add"):
            missing.append("data/add")
        if missing:
            return self._skip_full(f"the fixture is staged with data/add and computer-exec; the driver lacks {missing}")
        try:
            try:
                problem = await self._stage()
            except Exception as e:
                problem = f"{type(e).__name__}: {e}"
            if problem:
                return self._skip_full(f"the fixture didn't start: {problem}")
            for check, section in (("P9", self._security), ("P3", self._geometry), ("P4", self._effects),
                                   ("P5", self._unchanged), ("P6", self._typing), ("P7", self._scroll_and_drag),
                                   ("X", self._pointer_input), ("X", self._data_add), ("X", self._recording)):
                await self._guarded(check, section)
        finally:
            await self._unstage()

    # Recording

    async def check(self, check: str, name: str, test: Test, *extensions: str) -> bool:
        try:
            problem = await test()
        except Exception as e:
            problem = f"{type(e).__name__}: {e}"
        self._record(check, name, "failed" if problem else "passed", problem or "", extensions)
        return problem is None

    async def _guarded(self, check: str, section: Callable[[], Awaitable[None]]) -> None:
        """Run a section of checks; one that raises outside its checks is a failed check of its own."""
        try:
            await section()
        except Exception as e:
            self._record(check, f"{section.__name__.strip('_').replace('_', ' ')} could run", "failed",
                         f"{type(e).__name__}: {e}", ())

    def _record(self, check: str, name: str, status: str, detail: str, extensions: tuple[str, ...]) -> None:
        on = [COMPUTER, *extensions] if check.startswith("P") else list(extensions)
        self.results.append(CheckResult(check=check, name=name, status=status, detail=detail[:500], extensions=on))

    def _skip_full(self, why: str) -> None:
        for check, name in FULL_CHECKS:
            self._record(check, name, "unverified", why, ())

    # Talking to the driver

    async def tool(self, name: str, args: dict | None = None) -> CallToolResult:
        """Call ``name``, its points in frame pixels unless ``args`` names a space, whatever the driver's default."""
        args = args or {}
        return await self.session.call_tool(name, {"space": "frame", **args} if name in POINT_TOOLS else args)

    async def shot(self, **args) -> tuple[CallToolResult, object]:
        result = await self.tool("computer_screenshot", args)
        if result.isError:
            raise AssertionError(f"computer_screenshot failed: {result.content[0].text}")
        return result, image(result)

    async def act(self, name: str, args: dict) -> None:
        result = await self.tool(name, args)
        if result.isError:
            raise AssertionError(f"{name} {args} failed: {result.content[0].text}")

    async def state(self) -> dict:
        out = await self.driver.run("cat", f"{STAGE}/state/state.json")
        return json.loads(out["stdout"])

    async def until(self, holds: Callable[[dict], bool]) -> tuple[bool, dict]:
        """Whether the fixture's state comes to satisfy ``holds`` within SETTLE_S, and the state it last had."""
        deadline = time.monotonic() + SETTLE_S
        while True:
            state = await self.state()
            if holds(state) or time.monotonic() > deadline:
                return holds(state), state
            await asyncio.sleep(POLL_S)

    async def settled(self, holds: Callable[[dict], bool], show: Callable[[dict], object]) -> str | None:
        ok, state = await self.until(holds)
        return None if ok else f"the fixture shows {show(state)!r}"

    @staticmethod
    def refused(result: CallToolResult, code: str) -> str | None:
        data = result.structuredContent or {}
        if not result.isError:
            return f"it wasn't refused: {data.get('summary')!r}"
        if data.get("code") != code:
            return f"refused {data.get('code')!r}, not {code!r}: {data.get('message')}"
        if data.get("applied") != "no":
            return f"refused with applied {data.get('applied')!r}, not 'no'"
        return None

    # P0 to P2: readiness, the card, frames

    async def _ready(self) -> None:
        async def ready() -> str | None:
            if not self.driver.declares(COMPUTER_READY):
                return "the card doesn't declare computer-ready/v1"
            deadline = time.monotonic() + self.ready_timeout_s
            while True:
                status = await self.driver.invoke(COMPUTER_READY, "status")
                if status.get("ready"):
                    return None
                if time.monotonic() > deadline:
                    failing = [c for c in status.get("checks", []) if not c.get("ok")]
                    return f"not ready after {self.ready_timeout_s:g} s: {failing}"
                await asyncio.sleep(1)

        async def capture() -> str | None:
            await self.shot()
            return None

        await self.check("P0", "computer-ready reports ready", ready, COMPUTER_READY)
        await self.check("P0", "a screenshot works", capture)

    async def _card(self) -> None:
        tools = {t.name: t for t in (await self.session.list_tools()).tools}

        async def declared() -> str | None:
            return None if self.driver.declares(COMPUTER) else "the card doesn't declare computer/v1"

        async def minimum() -> str | None:
            missing = [n for n in MINIMUM_TOOLS if n not in tools]
            bare = [n for n in MINIMUM_TOOLS if n in tools and not (tools[n].inputSchema and tools[n].outputSchema)]
            if missing:
                return f"missing {missing}"
            return f"listed without an input or output schema: {bare}" if bare else None

        async def groups() -> str | None:
            problems = []
            for uri, names in ((COMPUTER_POINTER, GROUP_TOOLS["pointer"]), (COMPUTER_KEYBOARD, GROUP_TOOLS["keyboard"])):
                listed = [n for n in names if n in tools]
                if self.driver.declares(uri) and len(listed) != len(names):
                    problems.append(f"{uri} is declared without {sorted(set(names) - set(listed))}")
                if not self.driver.declares(uri) and listed:
                    problems.append(f"{listed} are listed without {uri}")
            return "; ".join(problems) or None

        await self.check("P1", "the card declares computer/v1", declared)
        await self.check("P1", "the seven minimum tools are listed with their schemas", minimum)
        await self.check("P1", "each group's tools come with its extension", groups)

    async def _frame(self) -> None:
        async def native() -> str | None:
            result, img = await self.shot()
            data, display = result.structuredContent, result.structuredContent["display"]
            want = (round(display["width"] * display["scale"]), round(display["height"] * display["scale"]))
            if img.size != (data["width"], data["height"]) or img.size != want:
                return f"the image is {img.size}, the frame says {(data['width'], data['height'])}, the display {want}"
            return None

        async def crop() -> str | None:
            result, img = await self.shot()
            w, h = img.size
            box = [w // 4, h // 4, w // 4 + w // 3, h // 4 + h // 3]
            cropped = await self.tool("computer_screenshot", {"region": box, "frame_id": result.structuredContent["frame_id"]})
            if cropped.isError:
                return cropped.content[0].text
            part = image(cropped)
            if part.size != (box[2] - box[0], box[3] - box[1]):
                return f"a {box} region came back {part.size}"
            if cropped.structuredContent.get("parent_frame_id") != result.structuredContent["frame_id"]:
                return f"its parent_frame_id is {cropped.structuredContent.get('parent_frame_id')!r}"
            points = [(0, 0), (part.width - 1, part.height - 1), (part.width // 2, part.height // 2)]
            wrong = [p for p in points if part.getpixel(p) != img.getpixel((box[0] + p[0], box[1] + p[1]))]
            return f"its pixels at {wrong} aren't its frame's" if wrong else None

        async def jpeg() -> str | None:
            result = await self.tool("computer_screenshot", {"format": "jpeg", "quality": 60})
            mime = next(c.mimeType for c in result.content if c.type == "image")
            return None if mime == "image/jpeg" else f"it answered {mime}"

        await self.check("P2", "a screenshot is native: its image, frame and display agree", native)
        await self.check("P2", "a region comes back as a native crop of its frame", crop)
        await self.check("P2", "format jpeg answers a JPEG", jpeg)

    # P5: refusals

    async def _refusal_cases(self) -> list[tuple[str, str, dict, str]]:
        """The calls a driver must refuse, aimed at a fresh frame, with the code each is refused with."""
        result, img = await self.shot()
        frame, (w, _h) = result.structuredContent["frame_id"], img.size
        over = int(self.features.get("max_click_count", 1)) + 1
        return [
            ("a point outside the frame", "computer_click", {"x": w, "y": 0, "frame_id": frame}, "out_of_bounds"),
            ("an unknown frame", "computer_click", {"x": 1, "y": 1, "frame_id": "f999999999"}, "stale_frame"),
            ("a key that isn't a W3C key value", "computer_key", {"keys": ["ctrl", "s"]}, "invalid_argument"),
            ("control characters in text", "computer_type", {"text": "a\x1bb"}, "invalid_argument"),
            ("a click count over the declared maximum", "computer_click", {"x": 1, "y": 1, "frame_id": frame,
                                                                           "count": over}, "invalid_argument"),
        ]

    async def _refusals(self) -> None:
        for n, (name, _tool, _args, code) in enumerate(await self._refusal_cases()):
            async def refused(n=n, code=code) -> str | None:
                _name, tool, args, _code = (await self._refusal_cases())[n]
                return self.refused(await self.tool(tool, args), code)
            await self.check("P5", f"{name} is refused {code}", refused)

    # The extensions' own methods

    async def _display(self) -> None:
        if not self.driver.declares(COMPUTER_DISPLAY):
            return

        async def get() -> str | None:
            got = await self.driver.invoke(COMPUTER_DISPLAY, "get")
            result, _ = await self.shot()
            frame = result.structuredContent["display"]
            seen = (frame["width"], frame["height"], frame["scale"])
            return None if (got["width"], got["height"], got["scale"]) == seen else f"get says {got}, frames {frame}"

        async def same() -> str | None:
            got = await self.driver.invoke(COMPUTER_DISPLAY, "get")
            again = await self.driver.invoke(COMPUTER_DISPLAY, "set", {"width": got["width"], "height": got["height"]})
            return None if (again["width"], again["height"]) == (got["width"], got["height"]) else f"set gave {again}"

        await self.check("X", "computer-display get matches the frames", get, COMPUTER_DISPLAY)
        if "set" in (self.driver.params(COMPUTER_DISPLAY).get("methods") or {}):
            await self.check("X", "computer-display set to the current mode keeps it", same, COMPUTER_DISPLAY)

    async def _spaces(self) -> None:
        if not self.driver.declares(COMPUTER_SPACES):
            return

        async def rewrite() -> str | None:
            before = await self.driver.invoke(COMPUTER_SPACES, "get")
            other = "norm" if before["default_space"] == "frame" else "frame"
            first = {t.name: t.description for t in (await self.session.list_tools()).tools}
            await self.driver.invoke(COMPUTER_SPACES, "set", {"default_space": other, "default_norm_scale": NORM})
            try:
                second = {t.name: t.description for t in (await self.session.list_tools()).tools}
                now = await self.driver.invoke(COMPUTER_SPACES, "get")
            finally:
                await self.driver.invoke(COMPUTER_SPACES, "set", {"default_space": before["default_space"],
                                                                  "default_norm_scale": before["default_norm_scale"]})
            if now["default_space"] != other:
                return f"set {other}, then get said {now}"
            changed = [n for n in POINT_TOOLS if n in first and first[n] != second.get(n)]
            return None if changed else "no point tool's description changed"

        await self.check("X", "computer-spaces set changes the default and the point tools' descriptions", rewrite,
                         COMPUTER_SPACES)

    async def _keyboard_settings(self) -> None:
        if not self.driver.declares(COMPUTER_KEYBOARD):
            return

        async def keep() -> str | None:
            got = await self.driver.invoke(COMPUTER_KEYBOARD, "get")
            again = await self.driver.invoke(COMPUTER_KEYBOARD, "set", {k: got[k] for k in got if k != "layout"})
            return None if again == got else f"set {got}, then got {again}"

        await self.check("X", "computer-keyboard set to its own settings keeps them", keep, COMPUTER_KEYBOARD)

    async def _exec(self) -> None:
        if not self.driver.declares(COMPUTER_EXEC):
            return
        cap = self.driver.params(COMPUTER_EXEC).get("max_output_bytes")

        async def output() -> str | None:
            out = await self.driver.run("sh", "-c", "echo probe; echo oops >&2; exit 3")
            want = {"exit_code": 3, "stdout": "probe\n", "stderr": "oops\n", "timed_out": False}
            return None if {k: out.get(k) for k in want} == want else f"it answered {out}"

        async def timeout() -> str | None:
            started = time.monotonic()
            out = await self.driver.run("sh", "-c", "sleep 30", timeout_s=1)
            took = time.monotonic() - started
            return None if out.get("timed_out") and took < 15 else f"after {took:.1f} s it answered {out}"

        async def once() -> str | None:
            request = {"argv": ["sh", "-c", "date +%s%N"], "request_id": f"{PROBE_ROLE}-once-{time.time_ns()}"}
            first = await self.driver.invoke(COMPUTER_EXEC, "run", request)
            again = await self.driver.invoke(COMPUTER_EXEC, "run", request)
            return None if first == again else f"it ran twice: {first} then {again}"

        async def capped() -> str | None:
            out = await self.driver.run("sh", "-c", f"head -c {cap + 1000} /dev/zero | tr '\\0' a")
            return None if out.get("truncated") and len(out["stdout"]) == cap else (
                f"{len(out['stdout'])} bytes came back, truncated {out.get('truncated')}")

        async def missing() -> str | None:
            try:
                out = await self.driver.run("agentenv-probe-no-such-program")
            except ExtensionCallError as e:
                return None if e.code else f"refused without a code: {e}"
            return f"it answered {out}"

        await self.check("X", "a command's exit code and output come back", output, COMPUTER_EXEC)
        await self.check("X", "a command past its timeout is stopped", timeout, COMPUTER_EXEC)
        await self.check("X", "a repeated request_id answers the first result", once, COMPUTER_EXEC)
        if isinstance(cap, int):
            await self.check("X", "output past max_output_bytes is cut there", capped, COMPUTER_EXEC)
        await self.check("X", "a program that isn't there is refused with a code", missing, COMPUTER_EXEC)

    async def _skill(self) -> None:
        if not self.driver.declares(SKILL_EXTENSION_URI):
            return

        async def written() -> str | None:
            got = await self.driver.invoke(SKILL_EXTENSION_URI, "get", {"name": "probe-skill"})
            if got.get("name") != "probe-skill" or not str(got.get("skill_md")).startswith("---\nname: probe-skill\n"):
                return f"it answered {got}"
            return None if got.get("description") else "its description is empty"

        await self.check("X", "the skill is written under the name it's asked for", written, SKILL_EXTENSION_URI)

    async def _data_get(self) -> None:
        if not self.driver.declares(INTAKE_EXTENSION_URI):
            return

        async def get() -> str | None:
            got = await self.driver.rpc("data/get")
            return None if isinstance((got.get("result") or {}).get("parts"), list) else f"it answered {got}"

        await self.check("X", "data/get answers its parts", get, *DATA_PLANE)

    async def _app_exports(self) -> None:
        for app in self.driver.card.get("children_environments") or []:
            if "data/get" not in (protocol.operations(app) or []):
                continue

            async def export(app: dict = app) -> str | None:
                got = await self.driver.rpc("data/get", card=app)
                return None if isinstance((got.get("result") or {}).get("parts"), list) else f"it answered {got}"

            await self.check("X", f"app {app.get('name')}'s data/get answers its parts through the driver", export,
                             *DATA_PLANE)

    async def _tool_access(self) -> None:
        if not (self.driver.declares(DISABLE_TOOL_EXTENSION_URI) and self.driver.declares(ENABLE_TOOL_EXTENSION_URI)):
            return

        async def per_role() -> str | None:
            role = f"{PROBE_ROLE}-{uuid.uuid4().hex[:8]}"
            hidden = {"role": role, "tools": ["computer_wait"]}
            await self.driver.invoke(DISABLE_TOOL_EXTENSION_URI, "disable", hidden)
            try:
                async with self.driver.session(role) as other:
                    listed = {t.name for t in (await other.list_tools()).tools}
                    refused = await other.call_tool("computer_wait", {"ms": 0})
                allowed = await self.tool("computer_wait", {"ms": 0})
            finally:
                await self.driver.invoke(ENABLE_TOOL_EXTENSION_URI, "enable", hidden)
            async with self.driver.session(role) as other:
                back = await other.call_tool("computer_wait", {"ms": 0})
            if "computer_wait" in listed:
                return "a disabled tool is still listed for its role"
            if not refused.isError or (refused.structuredContent or {}).get("code") != "permission_denied":
                return f"a disabled tool answered its role: {refused.content[0].text if refused.content else ''}"
            if allowed.isError or back.isError:
                return "the tool was refused to another role, or after it was enabled again"
            return None

        await self.check("X", "a tool disabled for a role is hidden from it and refused, and only to it", per_role,
                         DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI)

    async def _pointer(self) -> None:
        if not self.driver.declares(COMPUTER_POINTER):
            return

        async def listed() -> str | None:
            pointers = (await self.tool("computer_pointer_list")).structuredContent.get("pointers") or []
            return None if sum(p.get("default") is True for p in pointers) == 1 else f"it listed {pointers}"

        async def moved() -> str | None:
            result, img = await self.shot()
            frame, (x, y) = result.structuredContent["frame_id"], (img.width // 3, img.height // 3)
            await self.act("computer_pointer_move", {"x": x, "y": y, "frame_id": frame})
            where = (await self.tool("computer_pointer_position", {"frame_id": frame})).structuredContent
            return None if (where.get("x"), where.get("y"), where.get("inside_frame")) == (x, y, True) else (
                f"moved to ({x}, {y}); it reports {where}")

        await self.check("X", "computer_pointer_list names one default pointer", listed, COMPUTER_POINTER)
        await self.check("X", "the pointer goes where computer_pointer_move sends it", moved, COMPUTER_POINTER)

    def _grants(self) -> None:
        params = self.driver.params(COMPUTER_POINTER) if self.driver.declares(COMPUTER_POINTER) else {}
        if "leasing" in params:
            self._record("P8", "leases refuse other agents' input", "unverified", "the probe has no lease checks yet",
                         (COMPUTER_POINTER,))
        else:
            self._record("P8", "leases refuse other agents' input", "not_applicable", "the driver declares no leasing",
                         ())

    # full: the fixture

    async def _stage(self) -> str | None:
        source = resources.files(__package__).joinpath("fixture.py").read_bytes()
        added = await self.driver.add_file(f"{STAGE}/fixture.py", source)
        if "error" in added:
            return f"data/add refused it: {added['error'].get('message')}"
        await self.driver.run("sh", "-c", f"rm -rf {STAGE}/state")
        started = await self.driver.run(
            "sh", "-c", f"python3 {STAGE}/fixture.py {STAGE}/state > {STAGE}/fixture.log 2>&1 & echo $!")
        self.fixture_pid = started["stdout"].strip() or None
        deadline = time.monotonic() + FIXTURE_START_S
        while time.monotonic() < deadline:
            layout = await self.driver.run("cat", f"{STAGE}/state/layout.json")
            if layout.get("exit_code") == 0 and layout["stdout"].strip():
                self.layout = json.loads(layout["stdout"])
                return None
            state = await self.driver.run("cat", f"{STAGE}/state/state.json")
            if state.get("exit_code") == 0 and "error" in (failed := json.loads(state["stdout"] or "{}")):
                return failed["error"]
            await asyncio.sleep(0.5)
        log = await self.driver.run("tail", "-n", "5", f"{STAGE}/fixture.log")
        return f"no layout after {FIXTURE_START_S:g} s: {log['stdout'].strip() or log['stderr'].strip()}"

    async def _unstage(self) -> None:
        if self.fixture_pid:
            await self.driver.run("kill", self.fixture_pid)
        await self.driver.run("rm", "-rf", STAGE)

    async def _base(self) -> tuple[str, float, object]:
        """A fresh frame of the fixture: its id, the display's scale, and its image."""
        result, img = await self.shot()
        return result.structuredContent["frame_id"], result.structuredContent["display"]["scale"], img

    def _at(self, box: str, scale: float, dx: float | None = None, dy: float | None = None) -> tuple[int, int]:
        """A point of a fixture widget in frame pixels: its centre, or ``dx``, ``dy`` in from its corner."""
        x, y, w, h = self.layout[box]
        return (math.floor((x + (w / 2 if dx is None else dx)) * scale),
                math.floor((y + (h / 2 if dy is None else dy)) * scale))

    async def _security(self) -> None:
        async def one_user() -> str | None:
            uid = (await self.driver.run("id", "-u"))["stdout"].strip()
            owner = (await self.driver.run("stat", "-c", "%u", f"{STAGE}/fixture.py"))["stdout"].strip()
            if uid == "0":
                return "commands run as root"
            return None if owner == uid else f"data/add wrote as uid {owner}, commands run as uid {uid}"

        await self.check("P9", "commands and data/add run as one unprivileged user", one_user, COMPUTER_EXEC,
                         *DATA_PLANE)

    async def _geometry(self) -> None:
        frame, scale, img = await self._base()
        markers = [(x, y, _rgb(colour)) for x, y, colour in self.layout["markers"]]

        async def pixels() -> str | None:
            wrong = [(x, y, colour, img.getpixel((math.floor(x * scale), math.floor(y * scale))))
                     for x, y, colour in markers
                     if img.getpixel((math.floor(x * scale), math.floor(y * scale))) != colour]
            return f"markers read wrong (x, y, want, got): {wrong}" if wrong else None

        async def norm() -> str | None:
            wrong = []
            for x, y, colour in markers:
                fx, fy = math.floor(x * scale), math.floor(y * scale)
                region = [(fx + 0.5) * NORM / img.width, (fy + 0.5) * NORM / img.height,
                          min((fx + 4.5) * NORM / img.width, NORM), min((fy + 4.5) * NORM / img.height, NORM)]
                cropped = await self.tool("computer_screenshot", {"region": region, "frame_id": frame, "space": "norm",
                                                                  "norm_scale": NORM})
                if cropped.isError:
                    return cropped.content[0].text
                if (got := image(cropped).getpixel((0, 0))) != colour:
                    wrong.append((x, y, colour, got))
            return f"norm regions start on the wrong pixel (x, y, want, got): {wrong}" if wrong else None

        await self.check("P3", "markers at odd and even pixels read back in frame pixels", pixels)
        if self.driver.declares(COMPUTER_SPACES):
            await self.check("P3", "norm points land on the same pixels", norm, COMPUTER_SPACES)

    async def _effects(self) -> None:
        frame, scale, img = await self._base()
        size = self.layout["cell"]
        cell = lambda row, col: self._at("grid", scale, col * size + size / 2, row * size + size / 2)  # noqa: E731
        buttons = self.features.get("buttons") or ["left"]

        async def frame_space() -> str | None:
            x, y = cell(2, 3)
            await self.act("computer_click", {"x": x, "y": y, "frame_id": frame})
            problem = await self.settled(lambda s: s["cells"].get("2,3") == 1, lambda s: s["cells"])
            if problem:
                return problem
            _, img = await self.shot()
            return None if img.getpixel((x, y)) == (255, 0, 0) else f"the next frame shows {img.getpixel((x, y))}"

        async def norm_space() -> str | None:
            x, y = cell(5, 5)
            await self.act("computer_click", {"x": (x + 0.5) * NORM / img.width, "y": (y + 0.5) * NORM / img.height,
                                              "frame_id": frame, "space": "norm", "norm_scale": NORM})
            return await self.settled(lambda s: s["cells"].get("5,5") == 1, lambda s: s["cells"])

        async def right() -> str | None:
            x, y = cell(0, 0)
            await self.act("computer_click", {"x": x, "y": y, "frame_id": frame, "button": "right"})
            return await self.settled(lambda s: s["right_clicks"] == 1, lambda s: s["right_clicks"])

        async def double() -> str | None:
            x, y = cell(7, 7)
            await self.act("computer_click", {"x": x, "y": y, "frame_id": frame, "count": 2})
            return await self.settled(lambda s: s["double_clicks"] >= 1, lambda s: s["double_clicks"])

        async def crop() -> str | None:
            gx, gy = self._at("grid", scale, 0, 0)
            gw, gh = math.floor(self.layout["grid"][2] * scale), math.floor(self.layout["grid"][3] * scale)
            cropped = await self.tool("computer_screenshot", {"region": [gx, gy, gx + gw, gy + gh], "frame_id": frame})
            if cropped.isError:
                return cropped.content[0].text
            inner = cropped.structuredContent["frame_id"]
            await self.act("computer_click", {"x": math.floor((4 * size + size / 2) * scale),
                                              "y": math.floor((6 * size + size / 2) * scale), "frame_id": inner})
            return await self.settled(lambda s: s["cells"].get("6,4") == 1, lambda s: s["cells"])

        async def together() -> str | None:
            targets = [cell(row, col) for row in (1, 3) for col in range(8)]
            results = await asyncio.gather(*(self.tool("computer_click", {"x": x, "y": y, "frame_id": frame})
                                             for x, y in targets))
            if any(r.isError for r in results):
                return f"{sum(r.isError for r in results)} of {len(results)} were refused"
            landed = lambda s: sum(s["cells"].get(f"{r},{c}", 0) >= 1 for r in (1, 3) for c in range(8))  # noqa: E731
            return await self.settled(lambda s: landed(s) == 16, lambda s: f"{landed(s)} of 16 cells")

        async def widgets() -> str | None:
            await self.act("computer_click", {"x": self._at("save", scale)[0], "y": self._at("save", scale)[1],
                                              "frame_id": frame})
            x0, y0, _, h = self.layout["subscribe"]
            await self.act("computer_click", {"x": math.floor((x0 + 12) * scale), "y": math.floor((y0 + h / 2) * scale),
                                              "frame_id": frame})
            return await self.settled(lambda s: s["saved_count"] == 1 and s["subscribed"],
                                      lambda s: (s["saved_count"], s["subscribed"]))

        await self.check("P4", "a click in frame space hits its cell, and the next frame shows it", frame_space)
        if self.driver.declares(COMPUTER_SPACES):
            await self.check("P4", "a click in norm space hits its cell", norm_space, COMPUTER_SPACES)
        if "right" in buttons:
            await self.check("P4", "a right click arrives as one", right)
        if int(self.features.get("max_click_count", 1)) >= 2:
            await self.check("P4", "a double click arrives as one", double)
        await self.check("P4", "a click in a crop's frame maps back to the screen", crop)
        await self.check("P4", "16 clicks sent at once all land", together)
        await self.check("P4", "a button and a checkbox take a click", widgets)

    async def _unchanged(self) -> None:
        async def nothing() -> str | None:
            before = await self.state()
            for _name, tool, args, _code in await self._refusal_cases():
                await self.tool(tool, args)
            await asyncio.sleep(SETTLE_S / 5)
            after = await self.state()
            changed = sorted(k for k in before if before[k] != after.get(k))
            return f"refused calls changed {changed}" if changed else None

        await self.check("P5", "refusals leave the fixture as it was", nothing)

    async def _typing(self) -> None:
        frame, scale, _ = await self._base()
        text = "Hello, Wörld! 東京 AbC"

        async def focus(box: str) -> None:
            x, y = self._at(box, scale)
            await self.act("computer_click", {"x": x, "y": y, "frame_id": frame})

        async def clear() -> None:
            await focus("name")
            await self.act("computer_key", {"keys": ["Control", "/"]})
            await self.act("computer_key", {"keys": ["Backspace"]})

        async def unicode() -> str | None:
            await clear()
            await self.act("computer_type", {"text": text})
            return await self.settled(lambda s: s["name"] == text, lambda s: s["name"])

        async def events() -> str | None:
            state = await self.state()
            missing = [k for k in ("H", "W", "space", "comma") if k not in state["keys"]]
            return f"no key events for {missing}" if missing else None

        async def chord() -> str | None:
            await focus("name")
            await self.act("computer_key", {"keys": ["Control", "/"]})
            await self.act("computer_key", {"keys": ["Backspace"]})
            return await self.settled(lambda s: s["name"] == "", lambda s: s["name"])

        async def repeat() -> str | None:
            await clear()
            await self.act("computer_type", {"text": "abcd"})
            await self.act("computer_key", {"keys": ["Backspace"], "repeat": 2})
            return await self.settled(lambda s: s["name"] == "ab", lambda s: s["name"])

        async def whitespace() -> str | None:
            await focus("notes")
            await self.act("computer_type", {"text": "one\ttwo\nthree"})
            return await self.settled(lambda s: s["notes"].endswith("one\ttwo\nthree"), lambda s: s["notes"])

        async def clipboard() -> str | None:
            state = await self.state()
            return None if state["clipboard"] == self.layout["clipboard"] else f"the clipboard is {state['clipboard']!r}"

        async def held() -> str | None:
            await clear()
            await self.act("computer_keyboard_down", {"keys": ["Shift"]})
            await self.act("computer_type", {"text": "abc"})
            await self.act("computer_keyboard_up", {})
            await self.act("computer_type", {"text": "d"})
            return await self.settled(lambda s: s["name"] == "ABCd", lambda s: s["name"])

        await self.check("P6", "Unicode and mixed-case text arrives whole", unicode)
        await self.check("P6", "typed text arrives as key events", events)
        await self.check("P6", "a chord selects all, and Backspace clears it", chord)
        await self.check("P6", "a repeated key presses that many times", repeat)
        await self.check("P6", "tabs and newlines arrive as typed", whitespace)
        await self.check("P6", "typing leaves the clipboard as it was", clipboard)
        if self.driver.declares(COMPUTER_KEYBOARD):
            await self.check("P6", "a key held by computer_keyboard_down stays down while text is typed", held,
                             COMPUTER_KEYBOARD)

    async def _scroll_and_drag(self) -> None:
        frame, scale, _ = await self._base()

        async def scroll_list() -> str | None:
            x, y = self._at("list", scale)
            await self.act("computer_scroll", {"x": x, "y": y, "direction": "down", "amount": 5, "frame_id": frame})
            return await self.settled(lambda s: s["scroll_top"] > 0, lambda s: s["scroll_top"])

        async def directions() -> str | None:
            x, y = self._at("wheel", scale)
            want = {"up": 2, "down": 3, "left": 2, "right": 1}
            for direction, amount in want.items():
                await self.act("computer_scroll", {"x": x, "y": y, "direction": direction, "amount": amount,
                                                   "frame_id": frame})
            return await self.settled(lambda s: s["wheel"] == want, lambda s: s["wheel"])

        async def drag() -> str | None:
            start, end, waypoint = (50, 50), (300, 250), (200, 100)
            points = [start, waypoint, end] if int(self.features.get("max_drag_points", 2)) >= 3 else [start, end]
            path = [list(self._at("pad", scale, x, y)) for x, y in points]
            await self.act("computer_drag", {"path": path, "duration_ms": 400, "frame_id": frame})
            ok, state = await self.until(lambda s: bool(s["drags"]))
            if not ok:
                return "no drag reached the pad"
            stroke = state["drags"][-1]
            if math.dist(stroke["start"], start) > 2 or math.dist(stroke["end"], end) > 2 or stroke["points"] <= 3:
                return f"the stroke was {stroke['start']} to {stroke['end']} in {stroke['points']} points"
            if len(points) == 3 and min(math.dist(p, waypoint) for p in stroke["path"]) > 8:
                return f"the stroke missed its waypoint {waypoint}"
            return None

        await self.check("P7", "a scroll moves a list", scroll_list)
        await self.check("P7", "scrolls up, down, left and right each arrive that many times", directions)
        await self.check("P7", "a drag starts and ends where it says, through its waypoints", drag)

    async def _pointer_input(self) -> None:
        if not self.driver.declares(COMPUTER_POINTER):
            return
        frame, scale, _ = await self._base()

        async def down_up() -> str | None:
            before = (await self.state())["saved_count"]
            x, y = self._at("save", scale)
            await self.act("computer_pointer_down", {"x": x, "y": y, "frame_id": frame})
            await self.act("computer_pointer_up", {})
            await self.act("computer_click", {})
            return await self.settled(lambda s: s["saved_count"] == before + 2, lambda s: s["saved_count"])

        await self.check("X", "pointer down and up, then a click at the pointer, each press a button", down_up,
                         COMPUTER_POINTER)

    async def _data_add(self) -> None:
        async def picked_up() -> str | None:
            message = f"hello from data/add {time.time_ns()}"
            added = await self.driver.add_file(f"{STAGE}/state/inbox.txt", message.encode())
            if "error" in added:
                return f"data/add refused it: {added['error'].get('message')}"
            return await self.settled(lambda s: s["loaded"] == message, lambda s: s["loaded"])

        await self.check("X", "a file data/add writes is there for an app to read", picked_up, *DATA_PLANE)

    async def _recording(self) -> None:
        if not self.driver.declares(COMPUTER_RECORDING):
            return
        params = self.driver.params(COMPUTER_RECORDING)
        fps = min(RECORD_FPS, (params.get("fps") or {}).get("max") or RECORD_FPS)
        made: list[str] = []
        state: dict = {}

        async def start(request_id: str) -> dict:
            started = await self.driver.invoke(COMPUTER_RECORDING, "start", {"request_id": request_id, "fps": fps})
            if started["recording_id"] not in made:
                made.append(started["recording_id"])
            return started

        async def recorded() -> str | None:
            request_id = f"{PROBE_ROLE}-recording-{uuid.uuid4().hex[:8]}"
            started = await start(request_id)
            state["joined"] = await start(request_id) == started
            await asyncio.sleep(RECORD_S)
            manifest = await self.driver.invoke(COMPUTER_RECORDING, "end", {"recording_id": started["recording_id"]})
            return await self._segments_match(manifest, fps)

        async def joined() -> str | None:
            return None if state.get("joined") else "a repeated request_id started a second recording"

        async def limited() -> str | None:
            most = params.get("max_concurrent")
            if not isinstance(most, int):
                return "the card declares no max_concurrent"
            for _ in range(most):
                await start(f"{PROBE_ROLE}-limit-{uuid.uuid4().hex[:8]}")
            try:
                extra = await start(f"{PROBE_ROLE}-limit-{uuid.uuid4().hex[:8]}")
            except ExtensionCallError as e:
                return None if e.code == "limit_exceeded" else f"refused {e.code!r}, not 'limit_exceeded': {e.detail}"
            return f"recording {most + 1} of at most {most} started: {extra['recording_id']}"

        try:
            await self.check("X", "a recording's segments match their files, with no gap between them", recorded,
                             COMPUTER_RECORDING)
            await self.check("X", "a repeated request_id joins the first recording", joined, COMPUTER_RECORDING)
            await self.check("X", "a recording past max_concurrent is refused", limited, COMPUTER_RECORDING)
        finally:
            for recording_id in made:
                with contextlib.suppress(Exception):
                    await self.driver.invoke(COMPUTER_RECORDING, "end", {"recording_id": recording_id})
                await self.driver.run("rm", "-rf", f"{RECORDINGS_DIR}/{recording_id}")
            await self.driver.run("sh", "-c", f"rmdir {RECORDINGS_DIR} 2>/dev/null || true")

    async def _segments_match(self, wire: dict, fps: int) -> str | None:
        """What is wrong with a recording without a grant, read back from the computer, or None."""
        try:
            manifest = RecordingManifest.model_validate(wire)
        except ValidationError as e:
            return f"its manifest isn't a computer-recording/v1 manifest: {e}"
        segments = manifest.segments
        if manifest.ended_by != "end" or not segments:
            return f"it ended by {manifest.ended_by!r} with {len(segments)} segments"
        if unmeasured := [s.index for s in segments if not s.measured]:
            return f"segments {unmeasured} lack their times, size, frame count or digest"
        span = (_when(segments[-1].ended_at) - _when(segments[0].started_at)).total_seconds()
        frames = sum(s.frames for s in segments)
        if span < RECORD_S - 2 / fps or frames < fps * RECORD_S / 2:
            return f"{frames} frames over {span:.2f} s of a {RECORD_S:g} s recording at {fps} fps"
        for before, after in zip(segments, segments[1:]):
            gap = (_when(after.started_at) - _when(before.ended_at)).total_seconds()
            if gap > 2 / fps:
                return f"{gap:.2f} s between segments {before.index} and {after.index}"
        for segment in segments:
            path = f"{RECORDINGS_DIR}/{segment.path}"
            out = await self.driver.run("python3", "-c", READ_DIGEST, path)
            if out.get("exit_code") != 0:
                return f"segment {segment.index} couldn't be read back: {out.get('stderr', '')[-200:]}"
            file = json.loads(out["stdout"])
            if (file["size"], file["sha256"]) != (segment.bytes, segment.sha256):
                return f"segment {segment.index}'s file is {file['size']} bytes, sha256 {file['sha256'][:12]}; " \
                       f"the manifest says {segment.bytes}, {segment.sha256[:12]}"
            data = b""
            while len(data) < file["size"]:
                chunk = await self.driver.run("python3", "-c", READ_CHUNK, path, str(len(data)), str(CHUNK_BYTES))
                read = base64.b64decode(chunk["stdout"]) if chunk.get("exit_code") == 0 else b""
                if not read:
                    return f"segment {segment.index} stopped reading back at byte {len(data)} of {file['size']}: " \
                           f"{chunk.get('stderr', '')[-200:]}"
                data += read
            mp4 = read_mp4(data)
            span = (_when(segment.ended_at) - _when(segment.started_at)).total_seconds()
            if ((mp4["width"], mp4["height"], mp4["frames"]) != (segment.width, segment.height, segment.frames)
                    or abs(mp4["duration_s"] - span) > 2 / fps):
                return f"segment {segment.index}'s MP4 is {mp4}; the manifest says {segment.width}x" \
                       f"{segment.height}, {segment.frames} frames over {span:.2f} s"
        return None


def _when(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)
