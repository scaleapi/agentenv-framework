"""A website browser's recording, browser-recording/v1: a first frame, one after each burst of an agent's acting
actions once it settles, of the tab the burst last acted in, keyed to its entries, and a last frame, each uploaded as
it's taken; at the end its log's lines since the start and its manifest; a frame it couldn't take kept with why; a
start again joining its recording and an end again answering the same; and a frame waiting for its tab's page to
load, of the active tab once the agent's has closed."""
from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from agentenv_protocol.computer.errors import DeviceUnavailable, InvalidArgument
from agentenv_protocol.transfers import HttpPostPolicyGrant, WriteNamespaceGrant
from agentenv_protocol.website_browser import recording as browser_recording
from agentenv_protocol.website_browser import control as browser_control
from agentenv_protocol.website_browser.control import NUDGE_S, BrowserControl
from agentenv_protocol.website_browser.devtools import DevToolsError
from agentenv_protocol.website_browser.log import BrowserLog
from agentenv_protocol.website_browser.recording import BrowserRecordingManifest, BrowserRecordings


def _grant() -> dict:
    return WriteNamespaceGrant(
        root_path="agentenv-browser-recordings/r1", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        max_objects=100, max_object_bytes=1 << 20, max_total_bytes=1 << 24,
        write=HttpPostPolicyGrant(kind="http-post-policy", url="https://objects.example.test/upload",
                                  fields={"policy": "p"}, path_field="key", file_field="file"),
    ).model_dump(mode="json")


class Uploads:
    """Every upload, by grant root and path, in place of the object store."""

    def __init__(self) -> None:
        self.by_path: dict[str, bytes] = {}

    def uploader(self, grant):
        uploads = self

        class Uploader:
            async def upload(self, path, source):
                uploads.by_path[f"{grant.root_path}/{path}"] = source
        return Uploader()


class Control:
    """The browser's control as a recording asks it: each frame the tab asked for, else T0, or why there's none."""

    def __init__(self) -> None:
        self.asked, self.fail = [], None

    async def frame(self, target, load_wait_s):
        self.asked.append(target)
        if self.fail:
            raise DeviceUnavailable(self.fail)
        tab = target or "T0"
        return base64.b64encode(f"jpeg of {tab}".encode()).decode(), "jpg", tab, f"https://a.test/{tab}"


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    uploads, control = Uploads(), Control()
    monkeypatch.setattr(browser_recording, "NamespaceUploader", uploads.uploader)
    monkeypatch.setattr(browser_recording, "SETTLE_S", 0.05)
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    return log, control, uploads, BrowserRecordings(log, control)


def _acted(log: BrowserLog, via: str, method: str, tab: str, acts: bool = True) -> int:
    begun = log.begin({"via": via, "who": "ada", "method": method, "acts": acts, "tab": tab})
    log.finish(begun, ok=True)
    return begun["seq"]


@pytest.mark.asyncio
async def test_a_frame_follows_each_settled_burst_of_an_agents_actions_on_its_tab_keyed_to_its_entries(recorded):
    log, control, uploads, recordings = recorded
    started = await recordings.start("run-1:record", _grant())
    typed = [_acted(log, "cdp", "Input.insertText", "T1") for _ in range(3)]
    _acted(log, "cdp", "DOM.getDocument", "T1", acts=False)
    _acted(log, "agentenv", "navigate", "T1")
    await asyncio.sleep(0.2)
    clicked = _acted(log, "mcp", "browser_click", "T2")
    await asyncio.sleep(0.2)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    frames = [(f.kind, f.after, f.tab, f.path) for f in manifest.frames]
    assert frames == [("first", [], "T0", "frames/0000.jpg"), ("action", typed, "T1", "frames/0001.jpg"),
                      ("action", [clicked], "T2", "frames/0002.jpg"), ("last", [], "T2", "frames/0003.jpg")]
    assert all(f.uploaded and f.content_type == "image/jpeg" and f.url == f"https://a.test/{f.tab}"
               for f in manifest.frames)
    root = f"agentenv-browser-recordings/r1/{started['recording_id']}"
    assert uploads.by_path[f"{root}/frames/0001.jpg"] == b"jpeg of T1"
    logged = [json.loads(line)["seq"] for line in uploads.by_path[f"{root}/log.jsonl"].decode().splitlines()]
    assert logged == list(range(manifest.log_after + 1, manifest.log_last + 1)) and len(logged) == 6
    assert json.loads(uploads.by_path[f"{root}/manifest.json"]) == manifest.model_dump(mode="json")
    assert (manifest.ended_by, manifest.log_uploaded, manifest.missing_frames) == ("end", True, 0)


@pytest.mark.asyncio
async def test_a_burst_still_settling_at_the_end_gets_its_frame_before_the_last(recorded):
    log, control, uploads, recordings = recorded
    started = await recordings.start("run-1:record", _grant())
    seq = _acted(log, "cdp", "Input.dispatchMouseEvent", "T1")
    manifest = await recordings.end(started["recording_id"], _grant(), "teardown")

    assert [(f["kind"], f["after"]) for f in manifest["frames"]] == [("first", []), ("action", [seq]), ("last", [])]
    assert manifest["ended_by"] == "teardown"


@pytest.mark.asyncio
async def test_an_end_waits_for_entries_in_flight_writes_an_action_still_going_as_unfinished_and_lists_the_rest(
        recorded, monkeypatch):
    log, control, uploads, recordings = recorded
    monkeypatch.setattr(browser_recording, "LOG_WAIT_S", 0.3)
    started = await recordings.start("run-1:record", _grant(), frames=False)
    finishing = log.begin({"via": "cdp", "who": "ada", "method": "Page.navigate", "acts": True, "tab": "T1"})
    asyncio.get_running_loop().call_later(0.1, lambda: log.finish(finishing, ok=True))
    complete = await recordings.end(started["recording_id"])
    started = await recordings.start("run-2:record", _grant(), frames=False)
    going = log.begin({"via": "mcp", "who": "default", "tool": "browser_navigate", "acts": True, "tab": "T1"})
    after = _acted(log, "cdp", "Input.insertText", "T1")
    held = log.reserve()
    ended = await recordings.end(started["recording_id"])
    lines = [json.loads(line) for line in uploads.by_path[
        f"agentenv-browser-recordings/r1/{ended['recording_id']}/log.jsonl"].decode().splitlines()]

    assert complete["log_missing"] == [] and complete["log_last"] == finishing["seq"]
    assert ended["log_missing"] == [held] and [line["seq"] for line in lines] == [going["seq"], after]
    assert lines[0]["action"]["unfinished"] is True


@pytest.mark.asyncio
async def test_a_frame_the_browser_cant_take_is_kept_with_why_and_counts_as_missing(recorded):
    log, control, uploads, recordings = recorded
    started = await recordings.start("run-1:record", _grant())
    control.fail = "the tab painted nothing within 3 s"
    _acted(log, "cdp", "Input.dispatchMouseEvent", "T1")
    await asyncio.sleep(0.2)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert [f.uploaded for f in manifest.frames] == [True, False, False] and manifest.missing_frames == 2
    assert manifest.frames[1].error == "DeviceUnavailable: the tab painted nothing within 3 s"


@pytest.mark.asyncio
async def test_without_frames_a_recording_takes_none_and_still_uploads_its_log(recorded):
    log, control, uploads, recordings = recorded
    started = await recordings.start("run-1:record", _grant(), frames=False)
    _acted(log, "cdp", "Input.dispatchMouseEvent", "T1")
    await asyncio.sleep(0.2)
    manifest = await recordings.end(started["recording_id"])

    assert manifest["frames"] == [] and manifest["log_uploaded"] and control.asked == []


@pytest.mark.asyncio
async def test_a_start_again_joins_its_recording_an_end_again_answers_the_same_and_an_unknown_one_is_refused(recorded):
    log, control, uploads, recordings = recorded
    first = await recordings.start("run-1:record", _grant())
    again = await recordings.start("run-1:record", _grant())
    ended = await recordings.end(first["recording_id"])

    assert again == first and await recordings.end(first["recording_id"]) == ended
    with pytest.raises(InvalidArgument, match="no recording nope"):
        await recordings.end("nope")
    with pytest.raises(InvalidArgument, match="reason is one of"):
        await recordings.end(first["recording_id"], reason="later")
    with pytest.raises(InvalidArgument, match="write namespace grant"):
        await recordings.start("run-2:record", {"root_path": "x"})


@pytest.mark.asyncio
async def test_every_open_recording_ends_as_its_env_closes(recorded):
    log, control, uploads, recordings = recorded
    await recordings.start("run-1:a", _grant())
    await recordings.start("run-1:b", _grant(), frames=False)

    ended = await recordings.end_all("reset")

    assert [m["ended_by"] for m in ended] == ["reset", "reset"] and await recordings.end_all() == []


class Pipe:
    """The browser's pipe for a frame: each capture recorded with its params, its answer a JPEG's bytes."""

    def __init__(self) -> None:
        self.listeners, self.captured = [], []

    def listen(self, listener) -> None:
        self.listeners.append(listener)

    def emit(self, method: str, params: dict, session=None) -> None:
        for listener in self.listeners:
            listener({"method": method, "params": params, **({"sessionId": session} if session else {})})

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method == "Page.captureScreenshot":
            self.captured.append((session_id, params, asyncio.get_running_loop().time()))
            return {"data": base64.b64encode(b"jpeg").decode()}
        return {}


@pytest.mark.asyncio
async def test_a_frame_waits_for_its_tabs_page_to_load_then_captures_it_as_a_jpeg_and_a_closed_tabs_is_the_active():
    pipe = Pipe()
    control = BrowserControl(pipe, "agent", (1280, 800))
    pipe.emit("Target.targetCreated", {"targetInfo": {"targetId": "T1", "type": "page", "url": "about:blank"}})
    pipe.emit("Target.attachedToTarget", {"sessionId": "s-T1", "targetInfo": {"targetId": "T1", "type": "page"}})
    pipe.emit("Page.frameNavigated", {"frame": {"id": "T1", "url": "https://a.test/", "loaderId": "L1"}}, "s-T1")
    loop = asyncio.get_running_loop()
    loaded_at = loop.time() + 0.2
    loop.call_later(0.2, pipe.emit, "Page.loadEventFired", {}, "s-T1")

    data, extension, tab, _ = await control.frame("T1", load_wait_s=5)

    (session, params, captured_at), = pipe.captured
    assert (session, extension, tab, base64.b64decode(data)) == ("s-T1", "jpg", "T1", b"jpeg")
    assert params["format"] == "jpeg" and params["quality"] == 80 and captured_at >= loaded_at
    assert (await control.frame("closed", load_wait_s=0))[2] == "T1"


def _two_tabs() -> tuple[Pipe, BrowserControl]:
    pipe = Pipe()
    control = BrowserControl(pipe, "agent", (1280, 800))
    for tab in ("T1", "T2"):
        pipe.emit("Target.targetCreated", {"targetInfo": {"targetId": tab, "type": "page", "url": "about:blank"}})
        pipe.emit("Target.attachedToTarget", {"sessionId": f"s-{tab}", "targetInfo": {"targetId": tab, "type": "page"}})
    return pipe, control


@pytest.mark.asyncio
async def test_while_a_dialog_keeps_a_tab_from_painting_its_frame_is_the_last_it_took():
    pipe, control = _two_tabs()
    first = await control.frame("T1", load_wait_s=0)
    pipe.emit("Page.javascriptDialogOpening", {"type": "confirm", "message": "Delete?", "url": "https://a.test/"}, "s-T1")

    during = await control.frame("T1", load_wait_s=0)

    assert during[:3] == first[:3] == (base64.b64encode(b"jpeg").decode(), "jpg", "T1") and len(pipe.captured) == 1
    pipe.emit("Page.javascriptDialogOpening", {"type": "alert", "message": "Hi", "url": "https://a.test/"}, "s-T2")
    with pytest.raises(DeviceUnavailable, match="no capture before it"):
        await control.frame("T2", load_wait_s=0)


class BehindPipe(Pipe):
    """A pipe whose tab is behind the active one, as Chrome answers its captures: only once another arrives, then
    both."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting: list[asyncio.Future] = []

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method != "Page.captureScreenshot":
            return {}
        self.captured.append((session_id, params, asyncio.get_running_loop().time()))
        answer = asyncio.get_running_loop().create_future()
        self.waiting.append(answer)
        if len(self.waiting) > 1:
            for waiting in self.waiting:
                if not waiting.done():
                    waiting.set_result({"data": base64.b64encode(b"behind").decode()})
            self.waiting = []
        return await asyncio.wait_for(answer, timeout_s)


class RefusingPipe(Pipe):
    """A pipe whose browser refuses its first ``refusals`` captures with ``why``, as it does a tab between documents
    while a navigation swaps them, and answers the rest."""

    def __init__(self, refusals: int, why: str = "Page.captureScreenshot: Not attached to an active page") -> None:
        super().__init__()
        self.refusals, self.why = refusals, why

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method == "Page.captureScreenshot" and self.refusals:
            self.refusals -= 1
            self.captured.append((session_id, params, asyncio.get_running_loop().time()))
            raise DevToolsError(self.why)
        return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)


class BusyPipe(Pipe):
    """A pipe whose page is too busy to paint for ``busy_s``, every capture answering only after it."""

    def __init__(self, busy_s: float) -> None:
        super().__init__()
        self.busy_s = busy_s

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method == "Page.captureScreenshot":
            await asyncio.wait_for(asyncio.sleep(self.busy_s), timeout_s)
        return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)


def _one_tab(pipe: Pipe) -> BrowserControl:
    control = BrowserControl(pipe, "agent", (1280, 800))
    pipe.emit("Target.targetCreated", {"targetInfo": {"targetId": "T1", "type": "page", "url": "about:blank"}})
    pipe.emit("Target.attachedToTarget", {"sessionId": "s-T1", "targetInfo": {"targetId": "T1", "type": "page"}})
    return control


@pytest.mark.asyncio
async def test_a_frame_refused_while_its_tab_is_between_documents_is_asked_again_until_it_shows():
    pipe = RefusingPipe(refusals=2)

    data, _, tab, _ = await _one_tab(pipe).frame("T1", load_wait_s=0)

    assert (base64.b64decode(data), tab, len(pipe.captured)) == (b"jpeg", "T1", 3)


@pytest.mark.asyncio
async def test_a_tab_that_closes_while_its_frame_is_asked_again_is_framed_as_the_active_tab_a_dialog_holds():
    pipe = RefusingPipe(refusals=0)
    control = BrowserControl(pipe, "agent", (1280, 800))
    for tab in ("T1", "T2"):
        pipe.emit("Target.targetCreated", {"targetInfo": {"targetId": tab, "type": "page", "url": "about:blank"}})
        pipe.emit("Target.attachedToTarget", {"sessionId": f"s-{tab}", "targetInfo": {"targetId": tab, "type": "page"}})
    held = await control.frame("T1", load_wait_s=0)
    pipe.emit("Page.javascriptDialogOpening", {"type": "confirm", "message": "Leave?", "url": "https://a.test/"}, "s-T1")
    pipe.refusals = 1
    asyncio.get_running_loop().call_later(0.05, pipe.emit, "Target.detachedFromTarget", {"sessionId": "s-T2"})

    framed = await control.frame("T2", load_wait_s=0)

    assert framed[:3] == held[:3] and held[2] == "T1" and [s for s, _, _ in pipe.captured] == ["s-T1", "s-T2"]


@pytest.mark.asyncio
async def test_a_frame_whose_session_is_gone_is_taken_on_the_session_its_tab_was_attached_on_anew():
    pipe = RefusingPipe(refusals=1, why="Page.captureScreenshot: Session with given id not found")
    control = _one_tab(pipe)

    def reattach() -> None:
        pipe.emit("Target.detachedFromTarget", {"sessionId": "s-T1"})
        pipe.emit("Target.attachedToTarget", {"sessionId": "s-T1b", "targetInfo": {"targetId": "T1", "type": "page"}})
    asyncio.get_running_loop().call_later(0.05, reattach)

    _, _, tab, _ = await control.frame("T1", load_wait_s=0)

    assert tab == "T1" and [s for s, _, _ in pipe.captured] == ["s-T1", "s-T1b"]


@pytest.mark.asyncio
async def test_a_frame_whose_only_tab_is_attached_anew_only_after_a_retry_waits_for_it_within_its_budget():
    pipe = RefusingPipe(refusals=1, why="Page.captureScreenshot: Session with given id not found")
    control = _one_tab(pipe)
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, pipe.emit, "Target.detachedFromTarget", {"sessionId": "s-T1"})
    loop.call_later(0.6, pipe.emit, "Target.attachedToTarget",
                    {"sessionId": "s-T1b", "targetInfo": {"targetId": "T1", "type": "page"}})

    _, _, tab, _ = await control.frame("T1", load_wait_s=0)

    assert tab == "T1" and [s for s, _, _ in pipe.captured] == ["s-T1", "s-T1b"]


class HangingPipe(Pipe):
    """A pipe whose captures of the sessions in ``hanging`` never answer, as a tab's that closes mid-capture."""

    def __init__(self, *hanging: str) -> None:
        super().__init__()
        self.hanging = set(hanging)

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method == "Page.captureScreenshot" and session_id in self.hanging:
            self.captured.append((session_id, params, asyncio.get_running_loop().time()))
            await asyncio.sleep(timeout_s)
            raise asyncio.TimeoutError()
        return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)


@pytest.mark.asyncio
async def test_a_frame_whose_tab_closes_while_its_capture_waits_is_taken_of_the_tab_left_at_once():
    pipe = HangingPipe("s-T2")
    control = BrowserControl(pipe, "agent", (1280, 800))
    for tab in ("T1", "T2"):
        pipe.emit("Target.targetCreated", {"targetInfo": {"targetId": tab, "type": "page", "url": "about:blank"}})
        pipe.emit("Target.attachedToTarget", {"sessionId": f"s-{tab}", "targetInfo": {"targetId": tab, "type": "page"}})
    loop = asyncio.get_running_loop()
    loop.call_later(0.1, pipe.emit, "Target.detachedFromTarget", {"sessionId": "s-T2"})
    started = loop.time()

    _, _, tab, _ = await control.frame("T2", load_wait_s=0)

    assert tab == "T1" and pipe.captured[-1][0] == "s-T1" and loop.time() - started < 2


@pytest.mark.asyncio
async def test_a_frame_the_browser_refuses_for_another_reason_is_missing_at_once():
    pipe = RefusingPipe(refusals=5, why="Page.captureScreenshot: Internal error")

    with pytest.raises(DeviceUnavailable, match="Internal error"):
        await _one_tab(pipe).frame("T1", load_wait_s=0)
    assert len(pipe.captured) == 1


@pytest.mark.asyncio
async def test_a_page_too_busy_to_paint_has_the_frame_budget_to_show_and_no_longer(monkeypatch):
    monkeypatch.setattr(browser_control, "FRAME_CAPTURE_S", 0.6)
    monkeypatch.setattr(browser_control, "NUDGE_S", 0.1)

    data, _, _, _ = await _one_tab(BusyPipe(busy_s=0.3)).frame("T1", load_wait_s=0)
    assert base64.b64decode(data) == b"jpeg"
    with pytest.raises(DeviceUnavailable, match="painted nothing within 0.6 s"):
        await _one_tab(BusyPipe(busy_s=1.0)).frame("T1", load_wait_s=0)


@pytest.mark.asyncio
async def test_a_tab_behind_the_active_one_is_framed_once_a_second_capture_nudges_it():
    pipe = BehindPipe()
    control = BrowserControl(pipe, "agent", (1280, 800))
    pipe.emit("Target.targetCreated", {"targetInfo": {"targetId": "T1", "type": "page", "url": "about:blank"}})
    pipe.emit("Target.attachedToTarget", {"sessionId": "s-T1", "targetInfo": {"targetId": "T1", "type": "page"}})

    asked = asyncio.get_running_loop().time()
    data, _, tab, _ = await control.frame("T1", load_wait_s=0)

    (_, _, _), (_, _, second) = pipe.captured
    assert (base64.b64decode(data), tab) == (b"behind", "T1") and second - asked >= NUDGE_S


@pytest.mark.asyncio
async def test_a_tab_that_closes_while_its_page_loads_is_framed_as_the_active_tab():
    pipe, control = _two_tabs()
    pipe.emit("Page.frameNavigated", {"frame": {"id": "T2", "url": "https://a.test/pop", "loaderId": "L2"}}, "s-T2")
    asyncio.get_running_loop().call_later(0.1, pipe.emit, "Target.detachedFromTarget", {"sessionId": "s-T2"})

    _, _, tab, _ = await control.frame("T2", load_wait_s=5)

    assert tab != "T2" and [session for session, _, _ in pipe.captured] == [f"s-{tab}"]
