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
from typing import Any

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
    COMPUTER_PAGE,
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
from .page_fixture import QUESTION, TITLE, TWO, TWO_TITLE
from .report import CheckResult

MINIMUM_TOOLS = ("computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_type",
                 "computer_key", "computer_wait")
DATA_PLANE = (INTAKE_EXTENSION_URI, DATA_OBJECTS_EXTENSION_URI)
#: Where the fixture is staged and keeps its state, relative to the session user's home.
STAGE = ".agentenv-probe"
#: How the probe reads a staged file's owner: through the python3 the fixture runs on, the same on every OS.
OWNER = ("python3", "-c", "import os, sys; print(os.stat(sys.argv[1]).st_uid)")
#: Where a page driver's computer_upload takes files from, relative to the session user's home.
UPLOADS = "Uploads"
#: What the probe answers the page fixture's prompt with.
WORD = "tangerine"
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
#: How many times a check whose effect a busy desk can miss is tried before it fails: a desk starved through a drag
#: sees its motion as one event, and keys sent while the click that focuses a box waits go wherever focus was.
BUSY_DESK_ATTEMPTS = 2
RECORD_FPS = 5
#: Prints a file's size and SHA-256 as JSON, run with ``python3 -c`` on the computer.
READ_DIGEST = ("import hashlib,json,sys;d=open(sys.argv[1],'rb').read();"
               "print(json.dumps({'size':len(d),'sha256':hashlib.sha256(d).hexdigest()}))")
#: Prints ``length`` bytes of a file from ``offset``, base64, under computer-exec's output cap.
READ_CHUNK = ("import base64,sys;f=open(sys.argv[1],'rb');f.seek(int(sys.argv[2]));"
              "print(base64.b64encode(f.read(int(sys.argv[3]))).decode())")
CHUNK_BYTES = 512 << 10
#: The probe's commands on a Windows computer, which has no POSIX shell or tools: Python one-liners on the python3 the
#: fixture runs on, writing bytes, so output carries no CRLF, and taskkill and the account checks Windows has.
WIN_ECHO = ("python3", "-c", "import sys;sys.stdout.buffer.write(b'probe\\n');sys.stderr.buffer.write(b'oops\\n');"
            "sys.exit(3)")
WIN_SLEEP = ("python3", "-c", "import time;time.sleep(30)")
WIN_NOW = ("python3", "-c", "import time;print(time.time_ns())")
WIN_CAT = ("python3", "-c", "import sys;sys.stdout.buffer.write(open(sys.argv[1],'rb').read())")
WIN_TAIL = ("python3", "-c", "import sys;sys.stdout.buffer.write(b''.join(open(sys.argv[1],'rb').readlines()[-5:]))")
WIN_REMOVE = ("python3", "-c", "import os,shutil,sys;[shutil.rmtree(p,ignore_errors=True) if os.path.isdir(p) else "
              "os.path.exists(p) and os.remove(p) for p in sys.argv[1:]]")
WIN_RMDIR = ("python3", "-c", "import os,sys;p=sys.argv[1];os.path.isdir(p) and not os.listdir(p) and os.rmdir(p)")
#: Starts the fixture detached, its output to its log, in a process group of its own with no console window; its pid.
WIN_START = ("python3", "-c", "import subprocess,sys;log=open(sys.argv[3],'wb');print(subprocess.Popen("
             "[sys.executable,sys.argv[1],sys.argv[2]],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,"
             "creationflags=0x08000200).pid)")
#: A file's owner, its path in the command, since PowerShell takes what follows -Command as more of the command.
WIN_OWNER = "(Get-Acl -LiteralPath '{}').Owner"
#: The well-known SID of the local Administrators group, which an unprivileged user's token holds no enabled entry of.
ADMINISTRATORS = "S-1-5-32-544"
#: The checks only ``full`` runs, recorded unverified when the fixture can't be staged.
FULL_CHECKS = (("P3", "fixture colours land where the mapping says"), ("P4", "clicks show their effect"),
               ("P5", "refusals leave the fixture as it was"), ("P6", "typed and pressed keys arrive"),
               ("P7", "scrolls and drags arrive"), ("P9", "commands and data/add run as one unprivileged user"))

Test = Callable[[], Awaitable[str | None]]


class NotApplicable(Exception):
    """Raised by a test whose check doesn't apply to this driver, with why."""


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
        self.page = self.features.get("screen") == "page"
        platform = self.features.get("platform") or {}
        #: Whether the desk is the GTK one: on Wayland, where Tk runs only under XWayland, which a Wayland driver may
        #: not reach.
        self.gtk = not self.page and platform.get("display") == "wayland"
        self.fixture = "page_fixture.py" if self.page else "wayland_fixture.py" if self.gtk else "fixture.py"
        self.windows = platform.get("os") == "windows"
        #: The chord that selects a text box's text: Tk's Ctrl+/, which the Tk desk binds everywhere, or on the GTK
        #: desk GTK's own Ctrl+A.
        self.select_all = ["Control", "a"] if self.gtk else ["Control", "/"]
        self.desk = ""
        self.tab: str | None = None
        self.opened: list[str] = []
        self.upload: str | None = None

    async def basic(self) -> None:
        for check, section in (("P0", self._ready), ("P1", self._card), ("P2", self._frame), ("P5", self._refusals),
                               ("X", self._display), ("X", self._spaces), ("X", self._keyboard_settings),
                               ("X", self._exec), ("X", self._data_get), ("X", self._app_exports),
                               ("X", self._tool_access), ("X", self._pointer), ("X", self._skill),
                               ("X", self._control)):
            await self._guarded(check, section)
        self._grants()

    async def full(self) -> None:
        needs = (COMPUTER_EXEC, COMPUTER_PAGE) if self.page else (COMPUTER_EXEC,)
        missing = [uri for uri in needs if not self.driver.declares(uri)]
        if not (protocol.intake_declaration(self.driver.card) or {}).get("add"):
            missing.append("data/add")
        if missing:
            how = "data/add and computer-exec, and opened with computer-page/v1" if self.page else \
                "data/add and computer-exec"
            return self._skip_full(f"the fixture is staged with {how}; the driver lacks {missing}")
        try:
            try:
                problem = await self._stage()
            except Exception as e:
                problem = f"{type(e).__name__}: {e}"
            if problem:
                return self._skip_full(f"the fixture didn't start: {problem}")
            for check, section in (("P9", self._security), ("P3", self._geometry), ("P4", self._effects),
                                   ("P5", self._unchanged), ("P6", self._typing), ("P7", self._scroll_and_drag),
                                   ("X", self._pointer_input), ("X", self._data_add), ("X", self._recording),
                                   ("X", self._page_tools)):
                await self._guarded(check, section)
        finally:
            await self._unstage()

    # Recording

    async def check(self, check: str, name: str, test: Test, *extensions: str, attempts: int = 1) -> bool:
        """Run ``test`` up to ``attempts`` times until it passes; a pass after a failed try says what that try found."""
        tried: list[str] = []
        while True:
            try:
                problem = await test()
            except NotApplicable as e:
                self._record(check, name, "not_applicable", str(e), extensions)
                return True
            except Exception as e:
                problem = f"{type(e).__name__}: {e}"
            if problem is None or len(tried) + 1 == attempts:
                break
            tried.append(problem)
        detail = problem or (f"passed on try {len(tried) + 1} of {attempts}; before: {'; '.join(tried)}" if tried else "")
        self._record(check, name, "failed" if problem else "passed", detail, extensions)
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

    async def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        """The pixel at frame pixel (x, y) of a new frame."""
        return (await self.shot())[1].getpixel((x, y))

    async def act(self, name: str, args: dict) -> None:
        result = await self.tool(name, args)
        if result.isError:
            raise AssertionError(f"{name} {args} failed: {result.content[0].text}")

    async def state(self) -> dict:
        out = await self._cat(f"{STAGE}/state/state.json")
        return json.loads(out["stdout"])

    async def until(self, holds: Callable[[Any], bool],
                    read: Callable[[], Awaitable[Any]] | None = None) -> tuple[bool, Any]:
        """Whether what ``read`` gives, the fixture's state unless it's given, comes to satisfy ``holds`` within
        SETTLE_S, and what it last gave."""
        deadline = time.monotonic() + SETTLE_S
        while True:
            seen = await (read or self.state)()
            if holds(seen) or time.monotonic() > deadline:
                return holds(seen), seen
            await asyncio.sleep(POLL_S)

    async def settled(self, holds: Callable[[dict], bool], show: Callable[[dict], object]) -> str | None:
        ok, state = await self.until(holds)
        return None if ok else f"the fixture shows {show(state)!r}"

    async def page_now(self) -> dict:
        """The page a new frame names, or {} for a frame that names none."""
        return (await self.shot())[0].structuredContent.get("page") or {}

    async def showing(self, holds: Callable[[dict], bool]) -> str | None:
        ok, page = await self.until(holds, self.page_now)
        return None if ok else f"the frame's page is {page!r}"

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
            out = await self.driver.run(*(WIN_ECHO if self.windows else ("sh", "-c", "echo probe; echo oops >&2; exit 3")))
            want = {"exit_code": 3, "stdout": "probe\n", "stderr": "oops\n", "timed_out": False}
            return None if {k: out.get(k) for k in want} == want else f"it answered {out}"

        async def timeout() -> str | None:
            started = time.monotonic()
            out = await self.driver.run(*(WIN_SLEEP if self.windows else ("sh", "-c", "sleep 30")), timeout_s=1)
            took = time.monotonic() - started
            return None if out.get("timed_out") and took < 15 else f"after {took:.1f} s it answered {out}"

        async def once() -> str | None:
            request = {"argv": list(WIN_NOW if self.windows else ("sh", "-c", "date +%s%N")),
                       "request_id": f"{PROBE_ROLE}-once-{time.time_ns()}"}
            first = await self.driver.invoke(COMPUTER_EXEC, "run", request)
            again = await self.driver.invoke(COMPUTER_EXEC, "run", request)
            return None if first == again else f"it ran twice: {first} then {again}"

        async def capped() -> str | None:
            out = await self.driver.run(*(("python3", "-c", f"import sys;sys.stdout.buffer.write(b'a'*{cap + 1000})")
                                          if self.windows else ("sh", "-c", f"head -c {cap + 1000} /dev/zero | tr '\\0' a")))
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

    async def _control(self) -> None:
        if not self.driver.headers:
            return

        async def refused() -> str | None:
            for sent, without in (({}, "without"), (protocol.control_headers("not-its-token"), "with a wrong")):
                try:
                    await protocol.invoke_extension(self.driver.url, self.driver.card, COMPUTER_READY, method="status",
                                                    timeout=int(self.driver.timeout_s), headers=sent)
                except ExtensionCallError as e:
                    if e.response.status_code != 401:
                        return f"computer-ready status answered a call {without} control token {e.response.status_code}"
                    continue
                return f"computer-ready status answered a call {without} control token"
            return None

        await self.check("X", "a framework-only route refuses a call without the control token", refused)

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
        """The Fixture Desk started, or what stopped it: the Tk desk or on Wayland the GTK desk, either beside the layout
        they share (desk.py), or on a page, the HTML desk's server, with the desk opened in a tab of the probe's own."""
        for file in [self.fixture] if self.page else [self.fixture, "desk.py"]:
            source = resources.files(__package__).joinpath(file).read_bytes()
            added = await self.driver.add_file(f"{STAGE}/{file}", source)
            if "error" in added:
                return f"data/add refused it: {added['error'].get('message')}"
        if self.windows:
            await self.driver.run(*WIN_REMOVE, f"{STAGE}/state")
            started = await self.driver.run(*WIN_START, f"{STAGE}/{self.fixture}", f"{STAGE}/state", f"{STAGE}/fixture.log")
        else:
            await self.driver.run("sh", "-c", f"rm -rf {STAGE}/state")
            started = await self.driver.run(
                "sh", "-c", f"python3 {STAGE}/{self.fixture} {STAGE}/state > {STAGE}/fixture.log 2>&1 & echo $!")
        self.fixture_pid = started["stdout"].strip() or None
        deadline = time.monotonic() + FIXTURE_START_S
        while time.monotonic() < deadline:
            if self.page and self.tab is None and (problem := await self._open_desk()):
                return problem
            layout = await self._cat(f"{STAGE}/state/layout.json")
            if layout.get("exit_code") == 0 and layout["stdout"].strip():
                self.layout = json.loads(layout["stdout"])
                return None
            state = await self._cat(f"{STAGE}/state/state.json")
            if state.get("exit_code") == 0 and "error" in (failed := json.loads(state["stdout"] or "{}")):
                return failed["error"]
            await asyncio.sleep(0.5)
        log = await self.driver.run(*(WIN_TAIL if self.windows else ("tail", "-n", "5")), f"{STAGE}/fixture.log")
        return f"no layout after {FIXTURE_START_S:g} s: {log['stdout'].strip() or log['stderr'].strip()}"

    async def _open_desk(self) -> str | None:
        """Once the page fixture's server has its port, the desk opened in a new tab, which is the probe's."""
        port = await self._cat(f"{STAGE}/state/port")
        if port.get("exit_code") != 0 or not port["stdout"].strip():
            return None
        self.desk = f"http://127.0.0.1:{port['stdout'].strip()}/"
        opened = await self.tool("computer_tabs", {"action": "open", "url": self.desk})
        if opened.isError:
            return f"computer_tabs couldn't open it: {opened.content[0].text}"
        self.tab = next((t["tab"] for t in opened.structuredContent["tabs"] if t["active"]), None)
        self.opened += [self.tab] if self.tab else []
        return None if self.tab else f"the tab computer_tabs opened isn't active: {opened.structuredContent['tabs']}"

    async def _unstage(self) -> None:
        if self.opened:
            with contextlib.suppress(Exception):
                listed = await self.tool("computer_tabs", {"action": "list"})
                left = [t["tab"] for t in listed.structuredContent["tabs"]]
                for tab in [t for t in left if t in self.opened]:
                    if len(left) > 1:
                        await self.tool("computer_tabs", {"action": "close", "tab": tab})
                        left.remove(tab)
        if self.fixture_pid:
            await self.driver.run(*(("taskkill", "/F", "/T", "/PID") if self.windows else ("kill",)), self.fixture_pid)
        await self.driver.run(*(WIN_REMOVE if self.windows else ("rm", "-rf")), STAGE)
        if self.upload:
            await self.driver.run(*(WIN_REMOVE if self.windows else ("rm", "-f")), f"{UPLOADS}/{self.upload}")

    async def _cat(self, path: str) -> dict:
        """``path``'s bytes as the computer's command answers them."""
        return await self.driver.run(*(WIN_CAT if self.windows else ("cat",)), path)

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
        async def one_account() -> str | None:
            account = (await self.driver.run("whoami"))["stdout"].strip()
            owner = (await self.driver.run("powershell", "-NoProfile", "-Command",
                                           WIN_OWNER.format(f"{STAGE}/{self.fixture}")))["stdout"].strip()
            groups = (await self.driver.run("whoami", "/groups"))["stdout"]
            if any(ADMINISTRATORS in line and "Enabled group" in line for line in groups.splitlines()):
                return f"commands run as {account}, an administrator"
            return None if owner.lower() == account.lower() else (
                f"data/add wrote as {owner}, commands run as {account}")

        if self.windows:
            await self.check("P9", "commands and data/add run as one unprivileged user", one_account, COMPUTER_EXEC,
                             *DATA_PLANE)
            return

        async def one_user() -> str | None:
            uid = (await self.driver.run("id", "-u"))["stdout"].strip()
            owner = (await self.driver.run(*OWNER, f"{STAGE}/{self.fixture}"))["stdout"].strip()
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
            shows, pixel = await self.until(lambda p: p == (255, 0, 0), lambda: self.pixel(x, y))
            return None if shows else f"no frame within {SETTLE_S:g} s shows it: the last shows {pixel}"

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

        await self.check("P4", "a click in frame space hits its cell, and a frame shows it", frame_space)
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
            await self.act("computer_key", {"keys": self.select_all})
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
            await self.act("computer_key", {"keys": self.select_all})
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
            if pasted := state.get("pasted"):
                return f"typing arrived as a paste of {pasted!r}"
            if state["clipboard"] != self.layout["clipboard"]:
                return f"the clipboard is {state['clipboard']!r}"
            return None

        async def held() -> str | None:
            await clear()
            await self.act("computer_keyboard_down", {"keys": ["Shift"]})
            await self.act("computer_type", {"text": "abc"})
            await self.act("computer_keyboard_up", {})
            await self.act("computer_type", {"text": "d"})
            return await self.settled(lambda s: s["name"] == "ABCd", lambda s: s["name"])

        await self.check("P6", "Unicode and mixed-case text arrives whole", unicode)
        await self.check("P6", "typed text arrives as key events", events)
        await self.check("P6", "a chord selects all, and Backspace clears it", chord, attempts=BUSY_DESK_ATTEMPTS)
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
            before = (await self.state())["drags"]
            await self.act("computer_drag", {"path": path, "duration_ms": 400, "frame_id": frame})
            ok, state = await self.until(lambda s: s["drags"] != before)
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
        await self.check("P7", "a drag starts and ends where it says, through its waypoints", drag,
                         attempts=BUSY_DESK_ATTEMPTS)

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

    async def _page_tools(self) -> None:
        """The page tools against the HTML desk, each confirmed on the desk or in the page a new frame names; those
        that leave the desk run last."""
        if not self.page:
            return
        frame, scale, _ = await self._base()
        two = self.desk + TWO.lstrip("/")

        async def dialog() -> str | None:
            before = (await self.state())["saved_count"]
            await self.act("computer_click", {"x": self._at("ask", scale)[0], "y": self._at("ask", scale)[1],
                                              "frame_id": frame})

            async def seen() -> tuple[dict, dict]:
                return await self.page_now(), await self.state()

            ok, (page, state) = await self.until(lambda s: bool(s[0].get("dialog")) or s[1]["prompts"] > 0, seen)
            if not page.get("dialog"):
                if state["prompts"]:
                    raise NotApplicable(f"the browser answers dialogs itself: the prompt returned {state['prompted']!r}")
                return f"no dialog shows in the frame's page: {page!r}"
            if page["dialog"] != {"type": "prompt", "message": QUESTION, "default_text": ""}:
                return f"the frame's page shows the dialog as {page['dialog']!r}"
            x, y = self._at("save", scale)
            latest = (await self.shot())[0].structuredContent["frame_id"]
            if problem := self.refused(await self.tool("computer_click", {"x": x, "y": y, "frame_id": latest}),
                                       "dialog_open"):
                return f"a click while it's open: {problem}"
            await self.act("computer_dialog", {"accept": True, "text": WORD})
            return await self.settled(lambda s: s["prompted"] == WORD and s["saved_count"] == before,
                                      lambda s: {"prompted": s["prompted"], "saved_count": s["saved_count"]})

        async def upload() -> str | None:
            self.upload = f"{PROBE_ROLE}-{uuid.uuid4().hex[:8]}.txt"
            text = f"uploaded by the probe {time.time_ns()}"
            added = await self.driver.add_file(f"{UPLOADS}/{self.upload}", text.encode())
            if "error" in added:
                return f"data/add refused it: {added['error'].get('message')}"
            x, y = self._at("file", scale, dx=40)
            await self.act("computer_click", {"x": x, "y": y, "frame_id": frame})
            if problem := await self.showing(lambda p: p.get("file_chooser") is not None):
                return f"no file chooser shows: {problem}"
            outside = await self.tool("computer_upload", {"paths": [f"../{STAGE}/page_fixture.py"]})
            if problem := self.refused(outside, "invalid_argument"):
                return f"a file outside ~/{UPLOADS}: {problem}"
            await self.act("computer_upload", {"paths": [self.upload]})
            return await self.settled(lambda s: s["uploaded"] == {"name": self.upload, "content": text},
                                      lambda s: s["uploaded"])

        async def reload() -> str | None:
            loads = (await self.state())["loads"]
            await self.act("computer_history", {"direction": "reload"})
            return await self.settled(lambda s: s["loads"] == loads + 1, lambda s: f"{s['loads'] - loads} loads")

        async def navigate() -> str | None:
            await self.act("computer_navigate", {"url": two})
            return await self.showing(lambda p: p.get("url") == two and p.get("title") == TWO_TITLE)

        async def back_and_forth() -> str | None:
            for direction, url, title in (("back", self.desk, TITLE), ("forward", two, TWO_TITLE)):
                await self.act("computer_history", {"direction": direction})
                if problem := await self.showing(lambda p: p.get("url") == url and p.get("title") == title):
                    return f"after {direction}: {problem}"
            return None

        async def tabs() -> str | None:
            opened = await self.tool("computer_tabs", {"action": "open", "url": self.desk})
            if opened.isError:
                return f"open: {opened.content[0].text}"
            active = [t["tab"] for t in opened.structuredContent["tabs"] if t["active"]]
            if len(active) != 1 or active[0] == self.tab:
                return f"the opened tab isn't the one active: {opened.structuredContent['tabs']}"
            other = active[0]
            self.opened.append(other)
            if problem := await self.showing(lambda p: p.get("tab") == other and p.get("url") == self.desk):
                return f"after open: {problem}"
            await self.act("computer_tabs", {"action": "switch", "tab": self.tab})
            if problem := await self.showing(lambda p: p.get("tab") == self.tab):
                return f"after switching back: {problem}"
            closed = await self.tool("computer_tabs", {"action": "close", "tab": other})
            if closed.isError or any(t["tab"] == other for t in closed.structuredContent["tabs"]):
                return f"close: {closed.content[0].text if closed.isError else closed.structuredContent['tabs']}"
            return await self.showing(lambda p: p.get("tab") == self.tab)

        for name, test in (
                ("computer_dialog answers a prompt the frame's page shows, and input waits for it", dialog),
                (f"computer_upload gives the page's file chooser a file from ~/{UPLOADS}, and refuses one outside it",
                 upload),
                ("computer_history reload loads the page again", reload),
                ("computer_navigate opens a page, and the frame's page names it", navigate),
                ("computer_history goes back and forward through the tab's pages", back_and_forth),
                ("computer_tabs opens, switches to and closes a tab, and the frame shows the active one", tabs)):
            await self.check("X", name, test, COMPUTER_PAGE)

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
                await self.driver.run(*(WIN_REMOVE if self.windows else ("rm", "-rf")), f"{RECORDINGS_DIR}/{recording_id}")
            await self.driver.run(*((*WIN_RMDIR, RECORDINGS_DIR) if self.windows else
                                    ("sh", "-c", f"rmdir {RECORDINGS_DIR} 2>/dev/null || true")))

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
        if span < RECORD_S / 2 or frames < fps * RECORD_S / 2:
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
