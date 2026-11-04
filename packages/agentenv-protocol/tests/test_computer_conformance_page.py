"""The probe on a driver whose screen is a page: it stages the HTML desk in a tab of its own, verifies the input checks
and the page tools on it and leaves the tab it was given as it was, closing the tabs it opened even when a check
fails; without the page tools the input checks are unverified; a browser that answers dialogs itself makes the dialog
check not applicable; typing that pastes fails."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from probe_fake import FixtureDriver, PageFixtureDriver, served  # noqa: E402

from agentenv_protocol.computer import COMPUTER_PAGE, InvalidArgument, tabs, type_text  # noqa: E402
from agentenv_protocol.computer.conformance import ProbeReport, run_probe  # noqa: E402
from agentenv_protocol.computer.conformance import checks  # noqa: E402


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(checks, "RECORD_S", 0.8)
    return tmp_path


async def _probe(driver) -> ProbeReport:
    with served(driver) as url:
        return await run_probe(url, "full")


def _status(report: ProbeReport, start: str) -> tuple[str, str]:
    return next((c.status, c.detail) for c in report.checks if c.name.startswith(start))


@pytest.mark.asyncio
async def test_a_page_driver_passes_every_check_on_the_html_desk_and_the_probe_closes_its_own_tab(home):
    driver = PageFixtureDriver()
    report = await _probe(driver)

    failed = [f"{c.check} {c.name}: {c.detail}" for c in report.checks if c.status == "failed"]
    page = [c for c in report.checks if COMPUTER_PAGE in c.extensions]
    assert report.passed, failed
    assert report.counts()["unverified"] == 0 and report.extensions[COMPUTER_PAGE] == "verified"
    assert len(page) == 6 and {c.status for c in page} == {"passed"}
    assert [(t["id"], t["urls"]) for t in driver.tabs] == [("start", ["about:blank"])]
    assert not (home / ".agentenv-probe").exists() and not list((home / "Uploads").iterdir())
    assert ["kill", "4343"] in driver.commands and driver.desk.state["prompted"] == checks.WORD


@pytest.mark.asyncio
async def test_a_tab_the_probe_opened_is_closed_though_the_check_that_opened_it_failed():
    class NoSwitch(PageFixtureDriver):
        @tabs
        async def tab_action(self, action, tab, url):
            if action == "switch":
                raise InvalidArgument("this browser can't switch tabs")
            return await super().tab_action(action, tab, url)

    driver = NoSwitch()
    report = await _probe(driver)

    assert _status(report, "computer_tabs")[0] == "failed"
    assert [t["id"] for t in driver.tabs] == ["start"]


@pytest.mark.asyncio
async def test_a_page_driver_without_the_page_tools_leaves_the_input_checks_unverified_and_says_why():
    class NoPageTools(FixtureDriver):
        screen_kind = "page"

    report = await _probe(NoPageTools())

    unverified = {c.detail for c in report.checks if c.status == "unverified" and c.check != "P8"}
    assert report.passed and len(unverified) == 1
    assert "opened with computer-page/v1; the driver lacks ['urn:agentenv:computer-page/v1']" in unverified.pop()


@pytest.mark.asyncio
async def test_a_browser_that_answers_dialogs_itself_makes_the_dialog_check_not_applicable():
    report = await _probe(PageFixtureDriver(answers_dialogs=True))

    assert report.passed
    assert _status(report, "computer_dialog") == ("not_applicable",
                                                  "the browser answers dialogs itself: the prompt returned ''")


@pytest.mark.asyncio
async def test_typing_that_arrives_as_a_paste_fails_the_clipboard_check_on_a_page():
    class Pastes(PageFixtureDriver):
        @type_text
        async def type_in(self, text):
            await super().type_in(text)
            self.desk.state["pasted"].append(text)

    report = await _probe(Pastes())

    status, detail = _status(report, "typing leaves the clipboard as it was")
    assert status == "failed" and detail.startswith("typing arrived as a paste of ['Hello, Wörld! 東京 AbC'")
