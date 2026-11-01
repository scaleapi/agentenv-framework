"""The conformance probe against fake drivers on loopback: the fake driver with the fake fixture passes every check; a
driver without what full needs gets those checks unverified; a driver broken one way fails the check for that way; each
app a driver fronts that exports is asked for its export once; the report's verdicts and the CLI."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeScreen, MinimalDriver  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402
from PIL import ImageChops  # noqa: E402
from probe_fake import FixtureDriver, served  # noqa: E402

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
from agentenv_protocol.computer.conformance import checks  # noqa: E402
from agentenv_protocol.computer.conformance.client import Driver  # noqa: E402
from agentenv_protocol.computer.conformance.__main__ import main  # noqa: E402
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
async def test_a_driver_whose_default_space_is_norm_passes_at_full():
    report = await _probe(FixtureDriver(settings=ComputerSettings(default_space="norm")))
    assert report.passed, _failed(report)


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
    report = await _probe(MinimalDriver(FakeScreen(640, 400), platform=Platform(os="macos")))
    assert report.passed, _failed(report)
    unverified = [c for c in report.checks if c.status == "unverified"]
    assert {c.check for c in unverified} == {"P3", "P4", "P5", "P6", "P7", "P9"}
    assert "the driver lacks ['urn:agentenv:computer-exec/v1', 'data/add']" in unverified[0].detail
    assert "urn:agentenv:computer-exec/v1" not in report.extensions
    assert report.extensions["urn:agentenv:computer/v1"] == "verified"


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
    (ClicksOff, "a click in frame space hits its cell, and the next frame shows it"),
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


def test_the_cli_prints_each_check_writes_the_report_and_exits_by_the_result(tmp_path, capsys):
    out = tmp_path / "report.json"
    with served(FixtureDriver()) as url:
        passed = main([url, "--level", "basic", "--json", str(out)])
    with served(ExitZero()) as url:
        failed = main([url, "--level", "basic"])
    unreachable = main(["http://127.0.0.1:9", "--timeout", "2"])
    printed = capsys.readouterr()
    assert (passed, failed, unreachable) == (0, 1, 2)
    assert "PASS P0 computer-ready reports ready" in printed.out and "FAIL X a command's exit code" in printed.out
    assert "couldn't probe http://127.0.0.1:9" in printed.err
    assert ProbeReport.model_validate_json(out.read_text()).passed
