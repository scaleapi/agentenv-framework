"""The page group: a driver whose screen is a website browser's active tab serves computer-page/v1 and its tools, every
frame names the page it shows, computer_tabs answers with the tabs, and while a dialog is open on the page the input
it blocks is refused until computer_dialog answers it."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver, http_client, mcp_session  # noqa: E402

from agentenv_protocol.computer import (  # noqa: E402
    FileChooser,
    Image,
    PageDialog,
    PageState,
    Tab,
    capture,
    dialog,
    history,
    navigate,
    page_state,
    tabs,
    upload,
)

PAGE_TOOLS = {"computer_navigate", "computer_history", "computer_tabs", "computer_dialog", "computer_upload"}


class PageFake(FakeDriver):
    """A fake whose screen is a browser's page, with two tabs, the first active, and a log of the page calls."""

    def __init__(self) -> None:
        super().__init__()
        self.tabs = [Tab("a", "https://mail.test/", "Inbox", True), Tab("b", "https://docs.test/", "Docs")]
        self.open_dialog: PageDialog | None = None
        self.chooser: FileChooser | None = None
        self.calls: list = []
        self.captured: PageState | None = None

    @capture
    async def grab(self):
        image = await super().grab()
        if self.captured is None:
            return image
        return Image(image.width, image.height, image.tobytes(), "RGB", self.captured)

    @navigate
    async def go_to(self, url):
        self.calls.append(("navigate", url))

    @history
    async def step(self, direction):
        self.calls.append(("history", direction))

    @tabs
    async def tab_action(self, action, tab, url):
        self.calls.append(("tabs", action, tab, url))
        if action == "switch":
            self.tabs = [Tab(t.id, t.url, t.title, t.id == tab) for t in self.tabs]
        return self.tabs

    @dialog
    async def answer(self, accept, text):
        self.calls.append(("dialog", accept, text))
        self.open_dialog = None

    @upload
    async def choose(self, paths):
        self.calls.append(("upload", paths))

    @page_state
    async def page_now(self):
        active = next(t for t in self.tabs if t.active)
        return PageState(active.id, active.url, active.title, self.open_dialog, self.chooser)


@pytest.mark.asyncio
async def test_a_page_driver_declares_computer_page_and_serves_its_tools_and_a_desktop_driver_neither():
    async with http_client(PageFake()) as http:
        page = (await http.get("/.well-known/agent-env.json")).json()
    async with http_client(FakeDriver()) as http:
        desktop = (await http.get("/.well-known/agent-env.json")).json()

    def served(card):
        capabilities = card["capabilities"]
        return {e["uri"] for e in capabilities["extensions"]}, {t["name"] for t in capabilities["tools"]}

    page_uris, page_tools = served(page)
    desktop_uris, desktop_tools = served(desktop)
    assert "urn:agentenv:computer-page/v1" in page_uris and PAGE_TOOLS <= page_tools
    assert "urn:agentenv:computer-page/v1" not in desktop_uris and not PAGE_TOOLS & desktop_tools


@pytest.mark.asyncio
async def test_every_frame_names_the_page_it_shows_and_what_is_open_on_it():
    driver = PageFake()
    driver.chooser = FileChooser(multiple=True)
    async with mcp_session(driver) as session:
        shot = await session.call_tool("computer_screenshot", {})
        clicked = await session.call_tool("computer_click", {"x": 5, "y": 5, "screenshot": True})
    async with mcp_session(FakeDriver()) as session:
        desktop = await session.call_tool("computer_screenshot", {})

    page = {"tab": "a", "url": "https://mail.test/", "title": "Inbox", "file_chooser": {"multiple": True}}
    assert shot.structuredContent["page"] == page and clicked.structuredContent["frame"]["page"] == page
    assert shot.structuredContent["summary"].endswith(' on "Inbox" (https://mail.test/); a file chooser is open, '
                                                      "for files")
    assert ' on "Inbox" (https://mail.test/)' in clicked.structuredContent["summary"]
    assert "page" not in desktop.structuredContent


@pytest.mark.asyncio
async def test_the_page_tools_reach_the_driver_and_computer_tabs_answers_with_the_tabs():
    driver = PageFake()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        navigated = await session.call_tool("computer_navigate", {"url": "https://mail.test/sent"})
        await session.call_tool("computer_history", {"direction": "back"})
        switched = await session.call_tool("computer_tabs", {"action": "switch", "tab": "b"})
        refused = await session.call_tool("computer_tabs", {"action": "switch"})
        await session.call_tool("computer_upload", {"paths": ["cv.pdf"]})

    assert navigated.structuredContent["applied"] == "yes" and navigated.structuredContent["summary"].startswith(
        "Opened https://mail.test/sent")
    assert switched.structuredContent["tabs"] == [
        {"tab": "a", "url": "https://mail.test/", "title": "Inbox", "active": False},
        {"tab": "b", "url": "https://docs.test/", "title": "Docs", "active": True}]
    assert switched.structuredContent["summary"] == (
        'Switched to tab b; 2 open, the active one "Docs" (https://docs.test/)')
    assert refused.isError and refused.structuredContent["code"] == "invalid_argument"
    assert driver.calls == [("navigate", "https://mail.test/sent"), ("history", "back"), ("tabs", "switch", "b", None),
                            ("upload", ["cv.pdf"])]


@pytest.mark.asyncio
async def test_while_a_dialog_is_open_input_it_blocks_is_refused_until_computer_dialog_answers_it():
    driver = PageFake()
    driver.open_dialog = PageDialog("confirm", "Delete this message?")
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        refused = await session.call_tool("computer_click", {"x": 5, "y": 5})
        navigated = await session.call_tool("computer_navigate", {"url": "https://mail.test/"})
        listed = await session.call_tool("computer_tabs", {"action": "list"})
        answered = await session.call_tool("computer_dialog", {"accept": False})
        clicked = await session.call_tool("computer_click", {"x": 5, "y": 5})

    assert refused.structuredContent == {
        "code": "dialog_open", "applied": "no", "retryable": False,
        "message": 'a confirm dialog is open on the page: "Delete this message?"; answer it with computer_dialog first'}
    assert navigated.structuredContent["code"] == "dialog_open" and not listed.isError
    assert answered.structuredContent["summary"] == "Dismissed the dialog" and not clicked.isError
    assert ("dialog", False, None) in driver.calls and not any(entry[0] == "click" for entry in driver.screen.log[:-1])


@pytest.mark.asyncio
async def test_a_frame_names_the_page_its_capture_came_with_rather_than_one_read_after_it():
    driver = PageFake()
    driver.captured = PageState("b", "https://docs.test/", "Docs")
    async with mcp_session(driver) as session:
        shot = await session.call_tool("computer_screenshot", {})

    assert shot.structuredContent["page"] == {"tab": "b", "url": "https://docs.test/", "title": "Docs"}

