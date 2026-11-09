"""The conformance probe against fake drivers on loopback: the fake driver with the fake fixture passes every check; a
driver given a control token passes with it, refusing calls without it, and fails without it; a driver without what
full needs gets those checks unverified; a driver broken one way fails the check for that way; each app a driver
fronts that exports is asked for its export once; the report's verdicts and the CLI."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeScreen, MinimalDriver  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402
from PIL import ImageChops  # noqa: E402
from probe_fake import FakeDesk, FixtureDriver, served  # noqa: E402

from agentenv_protocol.computer import (  # noqa: E402
    ComputerSettings,
    ExecResult,
    Platform,
    ReadyCheck,
    capture,
    click,
    drag,
    key_down,
    run_in_session,
    scroll,
    type_text,
)
from agentenv_protocol.computer.conformance import CheckResult, ProbeReport, run_probe  # noqa: E402
from agentenv_protocol.computer import _linux  # noqa: E402
from agentenv_protocol.computer.conformance import checks  # noqa: E402
from agentenv_protocol.computer.conformance.client import Driver  # noqa: E402
from agentenv_protocol import CONTROL_TOKEN_ENV  # noqa: E402
from agentenv_protocol.tool_access import DISABLE_TOOL_EXTENSION_URI  # noqa: E402
from agentenv_protocol.types import EnvironmentCard  # noqa: E402


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(checks, "RECORD_S", 0.8)
    return tmp_path


async def _probe(driver, level="full", **kwargs) -> ProbeReport:
    with served(driver) as url:
        return await run_probe(url, level, **kwargs)


def _failed(report: ProbeReport) -> list[str]:
    return [f"{c.check} {c.name}: {c.detail}" for c in report.checks if c.status == "failed"]


def _status(report: ProbeReport, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


@pytest.mark.asyncio
async def test_the_fake_driver_passes_every_check_at_full_and_cleans_up(home):
    driver = FixtureDriver()
    report = await _probe(driver)
    assert report.passed, _failed(report)
    assert report.driver == "FixtureDriver" and report.level == "full"
    assert set(report.extensions.values()) == {"verified"} and len(report.extensions) == 13
    counts = report.counts()
    assert counts["failed"] == counts["unverified"] == 0 and counts["not_applicable"] == 1
    assert {c.check for c in report.checks} == {f"P{n}" for n in range(10)} | {"X"}
    assert not (home / ".agentenv-probe").exists() and not (home / ".agentenv-recordings").exists()
    assert ["kill", "4242"] in driver.commands


@pytest.mark.asyncio
async def test_a_driver_given_a_control_token_passes_with_it_and_refuses_calls_without_it(monkeypatch):
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "s3cret")

    report = await _probe(FixtureDriver(), "basic", token="s3cret")
    unauthorized = await _probe(FixtureDriver(), "basic")

    assert report.passed, _failed(report)
    assert _status(report, "a framework-only route refuses a call without the control token") == "passed"
    assert not unauthorized.passed and "401 unauthorized" in _failed(unauthorized)[0]


class StaleFrame(FixtureDriver):
    """Its first ``stale`` captures after a click are from before the click, as when an app records a click before it
    repaints."""

    def __init__(self, stale: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stale, self.left, self.before_click = stale, 0, None

    @click
    async def click_at(self, x, y, button, count, modifiers):
        self.before_click, self.left = self.screen.image.copy(), self.stale
        return await super().click_at(x, y, button, count, modifiers)

    @capture
    async def grab(self):
        if not self.left:
            return self.screen.image.copy()
        self.left -= 1
        return self.before_click.copy()


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", [lambda: FixtureDriver(settings=ComputerSettings(default_space="norm")), StaleFrame],
                         ids=["default-space-norm", "stale-first-frame"])
async def test_a_driver_whose_default_space_is_norm_or_whose_first_frame_after_a_click_is_stale_passes_at_full(driver):
    report = await _probe(driver())
    assert report.passed, _failed(report)


@pytest.mark.asyncio
async def test_frames_that_never_show_a_click_the_fixture_saw_fail_p4_with_the_last_pixel(monkeypatch):
    monkeypatch.setattr(checks, "SETTLE_S", 0.5)
    report = await _probe(StaleFrame(stale=10**6))
    failed = next(c for c in report.checks if c.name == "a click in frame space hits its cell, and a frame shows it")
    assert (failed.status, failed.detail) == ("failed", "no frame within 0.5 s shows it: the last shows (255, 255, 255)")


@pytest.mark.asyncio
async def test_a_tool_already_hidden_from_the_probes_role_stays_hidden():
    with served(FixtureDriver()) as url:
        driver = await Driver.connect(url, 10)
        await driver.invoke(DISABLE_TOOL_EXTENSION_URI, "disable", {"role": checks.PROBE_ROLE, "tools": ["computer_wait"]})
        report = await run_probe(url, "basic")
        async with driver.session(checks.PROBE_ROLE) as session:
            refused = await session.call_tool("computer_wait", {"ms": 0})
    assert report.passed, _failed(report)
    assert refused.isError and refused.structuredContent["code"] == "permission_denied"


@pytest.mark.asyncio
async def test_basic_needs_no_fixture(home):
    driver = FixtureDriver()
    report = await _probe(driver, "basic")
    assert report.passed, _failed(report)
    assert {c.check for c in report.checks} == {"P0", "P1", "P2", "P5", "P8", "X"}
    assert not driver.desk.started and not (home / ".agentenv-probe").exists()


class FrontsApps(FixtureDriver):
    """Fronts mail, whose card routes to the driver's own data plane, notes, which no app answers, and quiet, which
    doesn't export."""

    async def children_environments(self):
        return [EnvironmentCard(name=name, url=url, capabilities={"operations": operations})
                for name, url, operations in (("mail", "/agentenv", ["data/get"]),
                                              ("notes", "/svc/notes/agentenv", ["data/get"]),
                                              ("quiet", "/svc/quiet/agentenv", ["data/add"]))]


@pytest.mark.asyncio
async def test_each_app_a_driver_fronts_that_exports_is_asked_for_its_export_once(home):
    report = await _probe(FrontsApps(), "basic")

    assert [(c.name, c.status) for c in report.checks if c.name.startswith("app ")] == [
        ("app mail's data/get answers its parts through the driver", "passed"),
        ("app notes's data/get answers its parts through the driver", "failed")]


@pytest.mark.asyncio
async def test_a_driver_without_exec_or_data_add_passes_basic_and_leaves_full_unverified():
    report = await _probe(MinimalDriver(FakeScreen(640, 400), platform=Platform(os="freebsd")))
    assert report.passed, _failed(report)
    unverified = [c for c in report.checks if c.status == "unverified"]
    assert {c.check for c in unverified} == {"P3", "P4", "P5", "P6", "P7", "P9"}
    assert "the driver lacks ['urn:agentenv:computer-exec/v1', 'data/add']" in unverified[0].detail
    assert "urn:agentenv:computer-exec/v1" not in report.extensions
    assert report.extensions["urn:agentenv:computer/v1"] == "verified"


@pytest.mark.asyncio
async def test_a_windows_computer_is_probed_with_the_commands_windows_has(monkeypatch):
    """Its exec checks run Python one-liners rather than a POSIX shell; here they run on this machine's python3."""
    sent = []
    run = _linux.run_in_session

    async def recorded(argv, *args, **kwargs):
        sent.append(list(argv))
        return await run(argv, *args, **kwargs)

    monkeypatch.setattr(_linux, "run_in_session", recorded)
    report = await _probe(MinimalDriver(FakeScreen(640, 400), platform=Platform(os="windows", arch="amd64")), "basic")

    assert report.passed, _failed(report)
    exec_checks = [c for c in report.checks if "urn:agentenv:computer-exec/v1" in c.extensions]
    assert len(exec_checks) >= 4 and all(c.status == "passed" for c in exec_checks)
    assert list(checks.WIN_ECHO) in sent and list(checks.WIN_SLEEP) in sent and list(checks.WIN_NOW) in sent
    assert not [argv for argv in sent if argv[0] == "sh"]


class WaylandDesk(FakeDesk):
    """The GTK desk as the probe uses it: GTK's own Ctrl+A selects a text box's text, and Tk's Ctrl+/ does nothing,
    so a probe that sent it there would fail."""

    def press(self, keys: list[str]) -> None:
        if keys != ["Control", "/"]:
            super().press(["Control", "/"] if keys == ["Control", "a"] else keys)


class WaylandDriver(FixtureDriver):
    """A driver on a Wayland desktop, whose desk is the GTK one, started only from wayland_fixture.py."""

    def __init__(self) -> None:
        super().__init__(platform=Platform(os="linux", distribution="omarchy", version="4.5.0", arch="amd64",
                                           display="wayland"))
        self.desk = WaylandDesk(self)
        self.staged: list[str] = []

    def _fixture(self, argv):
        text = " ".join(argv)
        if f"python3 {checks.STAGE}/wayland_fixture.py" in text:
            self.staged = sorted(path.name for path in (Path.home() / checks.STAGE).glob("*.py"))
            self.desk.start()
            return ExecResult(0, b"4242\n")
        if f"python3 {checks.STAGE}/fixture.py" in text:
            return ExecResult(1, b"", b"Tk runs under XWayland here")
        if argv == [*checks.OWNER, f"{checks.STAGE}/wayland_fixture.py"]:
            return ExecResult(0, b"1000\n")
        return super()._fixture(argv)


@pytest.mark.asyncio
async def test_a_wayland_computer_is_probed_with_the_gtk_desk_beside_the_tk_desks_file_selecting_all_with_ctrl_a(home):
    driver = WaylandDriver()

    report = await _probe(driver)

    assert report.passed, _failed(report)
    assert driver.staged == ["desk.py", "wayland_fixture.py"]
    assert ["kill", "4242"] in driver.commands and not (home / checks.STAGE).exists()


class NeverReady(FixtureDriver):
    async def ready_checks(self):
        return [ReadyCheck("session", False, "no desktop session")]


class ClicksOff(FixtureDriver):
    @click
    async def click_at(self, x, y, button, count, modifiers):
        return await super().click_at(x + 60, y, button, count, modifiers)


class ShiftedCapture(FixtureDriver):
    @capture
    async def grab(self):
        return ImageChops.offset(self.screen.image, 1, 0)


class AsciiOnly(FixtureDriver):
    @type_text
    async def type_in(self, text):
        await super().type_in("".join(c for c in text if ord(c) < 128))


class SwappedScroll(FixtureDriver):
    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers):
        swapped = {"left": "right", "right": "left"}.get(direction, direction)
        await super().scroll_at(x, y, swapped, amount, modifiers)


class StraightDrag(FixtureDriver):
    @drag
    async def drag_along(self, path, button, duration_ms, modifiers):
        await super().drag_along([path[0], path[-1]], button, duration_ms, modifiers)


class NoHold(FixtureDriver):
    @key_down
    async def keys_down(self, keys):
        self.screen.log.append(("key_down", tuple(keys)))


class ExitZero(FixtureDriver):
    @run_in_session
    async def run(self, argv, cwd, env, timeout_s):
        result = await super().run(argv, cwd, env, timeout_s)
        return ExecResult(0, result.stdout, result.stderr, result.timed_out, result.truncated)


@pytest.mark.asyncio
@pytest.mark.parametrize("driver, broken", [
    (NeverReady, "computer-ready reports ready"),
    (ClicksOff, "a click in frame space hits its cell, and a frame shows it"),
    (ShiftedCapture, "markers at odd and even pixels read back in frame pixels"),
    (AsciiOnly, "Unicode and mixed-case text arrives whole"),
    (SwappedScroll, "scrolls up, down, left and right each arrive that many times"),
    (StraightDrag, "a drag starts and ends where it says, through its waypoints"),
    (NoHold, "a key held by computer_keyboard_down stays down while text is typed"),
    (ExitZero, "a command's exit code and output come back"),
])
async def test_a_driver_broken_one_way_fails_the_check_for_it(driver, broken, monkeypatch):
    monkeypatch.setattr(checks, "SETTLE_S", 0.5)
    report = await _probe(driver(), ready_timeout_s=1)
    failed = next(c for c in report.checks if c.name == broken)
    assert failed.status == "failed" and not report.passed
    assert failed.extensions and all(report.extensions[uri] == "failed" for uri in failed.extensions)


class NoFixture(FixtureDriver):
    def _fixture(self, argv):
        if argv[:1] == ["cat"] and argv[1].endswith("state.json"):
            return ExecResult(0, json.dumps({"error": "the screen is 800x600; the fixture needs 1280x800"}).encode())
        if "python3" in " ".join(argv):
            return ExecResult(0)
        return super()._fixture(argv)


class SilentFixture(FixtureDriver):
    def _fixture(self, argv):
        if "python3" in " ".join(argv):
            return ExecResult(0, b"4243\n")
        if argv[:1] == ["tail"]:
            return ExecResult(0, b"ModuleNotFoundError: No module named 'tkinter'\n")
        return super()._fixture(argv)


class ExecBreaks(FixtureDriver):
    def _fixture(self, argv):
        if f"rm -rf {checks.STAGE}/state" in " ".join(argv):
            raise RuntimeError("exec broke")
        return super()._fixture(argv)


@pytest.mark.asyncio
@pytest.mark.parametrize("driver, why", [
    (NoFixture, "the fixture didn't start: the screen is 800x600"),
    (SilentFixture, "no layout after 0.6 s: ModuleNotFoundError: No module named 'tkinter'"),
    (ExecBreaks, "the fixture didn't start: ExtensionCallError"),
])
async def test_a_fixture_that_doesnt_start_leaves_full_unverified_and_is_cleaned_up(driver, why, monkeypatch, home):
    monkeypatch.setattr(checks, "FIXTURE_START_S", 0.6)
    probed = driver()
    report = await _probe(probed)
    assert report.passed, _failed(report)
    unverified = [c for c in report.checks if c.status == "unverified"]
    assert len(unverified) == len(checks.FULL_CHECKS) and why in unverified[0].detail
    assert not (home / checks.STAGE).exists()
    assert (["kill", "4243"] in probed.commands) is (driver is SilentFixture)


@pytest.mark.parametrize("delta, steps", [(1, 1), (-3, -3), (120, 1), (-360, -3), (240, 2)],
                         ids=["aqua-up", "aqua-down-3", "scaled-up", "scaled-down-3", "scaled-up-2"])
def test_the_fixture_counts_a_wheel_events_steps_whether_tk_scales_them_by_120_or_not(delta, steps, monkeypatch):
    """Aqua's Tk 8.6 reports a step as a delta of 1; Windows, and Tk 9 everywhere, as 120."""
    assert _fixture(monkeypatch).wheel_steps(delta) == steps


@pytest.mark.parametrize("delta, steps", [(10, (False, 1)), (65526, (False, -1)), (65516, (False, -2)),
                                          (655360, (True, 1)), (-655360, (True, -1)), (0, None)],
                         ids=["up", "down", "down-2-coalesced", "left", "right", "still"])
def test_the_fixture_counts_a_scroll_tk_9_reports_in_pixels_by_the_lines_it_moves(delta, steps, monkeypatch):
    """The deltas Tk 9 on macOS gave cua-driver's scrolls: 10 pixels a line, two lines at once where macOS coalesced
    them."""
    assert _fixture(monkeypatch).touchpad_steps(delta) == steps


class _Root:
    """A Tk root stood in for: the attributes set on it in order, its bindings, the calls made to it, and what
    ``after`` was given to run, with its delay."""

    def __init__(self):
        self.set, self.calls, self.bound, self.timers = [], [], {}, []
        self.last = self

    def focus_lastfor(self):
        return self.last

    def attributes(self, name, value):
        self.set.append((name, value))

    def bind(self, sequence, handler, add=None):
        self.bound[sequence] = handler

    def after(self, ms, callback):
        self.timers.append((ms, callback))
        return f"after#{len(self.timers)}"

    def lift(self):
        self.calls.append("lift")

    def focus_force(self):
        self.calls.append("focus_force")


def test_the_fixture_on_x11_stays_maximized_above_other_windows_and_takes_the_top_back_once_one_covers_it(monkeypatch):
    """A window an app keeps above, as Typora's licence dialog is for its first seconds, otherwise takes the clicks the
    probe aims at the fixture."""
    fixture = _fixture(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(fixture.time, "monotonic", lambda: now[0])
    root = _Root()
    fixture.keep_on_top(root)
    seen = root.bound["<Visibility>"]

    seen(SimpleNamespace(widget="a child", state="VisibilityFullyObscured"))
    for_child = list(root.timers)
    seen(SimpleNamespace(widget=root, state="VisibilityPartiallyObscured"))
    seen(SimpleNamespace(widget=root, state="VisibilityPartiallyObscured"))
    (first, lift), = root.timers
    lift()
    lifted = (root.set[2:], list(root.calls))
    now[0] += 0.1
    seen(SimpleNamespace(widget=root, state="VisibilityFullyObscured"))
    second, again = root.timers[1]
    seen(SimpleNamespace(widget=root, state="VisibilityUnobscured"))
    again()

    assert root.set[:2] == [("-zoomed", True), ("-topmost", True)] and for_child == []
    assert first == 0 and lifted == ([("-topmost", False), ("-topmost", True)], ["lift", "focus_force"])
    assert second == round((fixture.LIFT_GAP_S - 0.1) * 1000) and root.calls == ["lift", "focus_force"]


def test_the_fixture_gives_the_focus_back_to_the_text_box_that_had_it_when_it_takes_the_top_back(monkeypatch):
    """A window an app opens during a typing check would otherwise leave the probe's keys with the fixture's root."""
    fixture = _fixture(monkeypatch)
    monkeypatch.setattr(fixture.time, "monotonic", lambda: 100.0)
    root, name = _Root(), _Root()
    root.last = name
    fixture.keep_on_top(root)

    root.bound["<Visibility>"](SimpleNamespace(widget=root, state="VisibilityPartiallyObscured"))
    (_, lift), = root.timers
    lift()

    assert root.calls == ["lift"] and name.calls == ["focus_force"]


def _fixture(monkeypatch):
    """The fixture module, which runs in the computer, read with Tk stood in for, as this Python may have none, and
    with desk.py importable as it is beside the fixture in the computer."""
    monkeypatch.setitem(sys.modules, "tkinter", ModuleType("tkinter"))
    monkeypatch.syspath_prepend(str(Path(checks.__file__).parent))
    monkeypatch.delitem(sys.modules, "agentenv_protocol.computer.conformance.fixture", raising=False)
    return importlib.import_module("agentenv_protocol.computer.conformance.fixture")


def test_the_gtk_desk_starts_on_a_computer_without_tk(monkeypatch):
    gi = ModuleType("gi")
    gi.require_version = lambda namespace, version: None
    gi.repository = SimpleNamespace(Gdk=None, GLib=None, Gtk=None)
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", gi.repository)
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.syspath_prepend(str(Path(checks.__file__).parent))
    for name in ("desk", "wayland_fixture"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    gtk_desk = importlib.import_module("wayland_fixture")

    assert gtk_desk.AT["grid"] == (400, 160)


def test_a_refusal_must_carry_the_code_and_applied_no():
    def result(error: bool, **data) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text="x")], structuredContent=data, isError=error)

    assert "wasn't refused" in checks.Probe.refused(result(False, summary="Clicked"), "out_of_bounds")
    assert "not 'out_of_bounds'" in checks.Probe.refused(result(True, code="internal"), "out_of_bounds")
    assert "applied 'unknown'" in checks.Probe.refused(result(True, code="out_of_bounds", applied="unknown"),
                                                      "out_of_bounds")
    assert checks.Probe.refused(result(True, code="out_of_bounds", applied="no"), "out_of_bounds") is None


def test_declared_leasing_is_unverified_until_the_probe_checks_leases():
    driver = SimpleNamespace(params=lambda uri: {"leasing": {}}, declares=lambda uri: True)
    probe = checks.Probe(driver, None, 1)
    probe._grants()
    assert probe.results[0].status == "unverified" and probe.results[0].check == "P8"


def test_an_extension_is_failed_by_any_check_verified_by_a_pass_and_unverified_otherwise():
    results = [CheckResult(check="X", name="a", status="passed", extensions=["e1", "e2"]),
               CheckResult(check="X", name="b", status="failed", extensions=["e2"]),
               CheckResult(check="X", name="c", status="unverified", extensions=["e3"])]
    report = ProbeReport.of("basic", "d", results, ["e1", "e2", "e3", "e4"])
    assert report.extensions == {"e1": "verified", "e2": "failed", "e3": "unverified", "e4": "unverified"}
    assert not report.passed and report.counts() == {"passed": 1, "failed": 1, "unverified": 1, "not_applicable": 0}


def test_the_cli_prints_each_check_writes_the_report_and_exits_by_the_result(tmp_path):
    out = tmp_path / "report.json"
    with served(FixtureDriver()) as url:
        passed = _cli(url, "--level", "basic", "--json", str(out))
    with served(ExitZero()) as url:
        failed = _cli(url, "--level", "basic")
    unreachable = _cli("http://127.0.0.1:9", "--timeout", "2")

    assert (passed.returncode, failed.returncode, unreachable.returncode) == (0, 1, 2)
    assert "PASS P0 computer-ready reports ready" in passed.stdout
    assert "FAIL X a command's exit code" in failed.stdout
    assert "couldn't probe http://127.0.0.1:9" in unreachable.stderr
    assert ProbeReport.model_validate_json(out.read_text()).passed


def _cli(*args: str) -> subprocess.CompletedProcess:
    """The conformance CLI run as its users run it, in a process of its own, so nothing in this one can take a line it
    prints."""
    return subprocess.run([sys.executable, "-m", "agentenv_protocol.computer.conformance", *args],
                          capture_output=True, text=True, timeout=180)
