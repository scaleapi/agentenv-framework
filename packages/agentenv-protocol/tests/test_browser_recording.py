"""A website browser's recording, browser-recording/v1, one per context: a video of the tab the context's agent is on,
following it to a tab it acts on, to a popup that tab opens, and back once one closes, by screencast, by screenshots
while the tab sends none, blank while the context has none, with nothing of another context in it; its context's log
lines; the browser's whole network log as HAR; frames after each burst of an agent's actions, keyed to its entries;
a context an agent makes over an address scoped to the recorded one recorded with it; and a recording that ends with
its lease. Then the browser's control, as recordings ask it: a frame waiting for its tab's page to load, of the active
tab once the agent's has closed, and screencasts each heard by everything that follows its tab."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from datetime import datetime, timedelta, timezone

import av
import pytest
import pytest_asyncio
from PIL import Image

from agentenv_protocol.computer import recording as computer_recording
from agentenv_protocol.computer.errors import DeviceUnavailable, InvalidArgument, TargetUnavailable, Unsupported
from agentenv_protocol.transfers import HttpPostPolicyGrant, WriteNamespaceGrant
from agentenv_protocol.website_browser import control as browser_control
from agentenv_protocol.website_browser import recording as browser_recording
from agentenv_protocol.website_browser import stream as browser_stream
from agentenv_protocol.website_browser.contexts import Contexts
from agentenv_protocol.website_browser.control import NUDGE_S, BrowserControl
from agentenv_protocol.website_browser.devtools import DevToolsError
from agentenv_protocol.website_browser.log import BrowserLog
from agentenv_protocol.website_browser.network import NetworkLog
from agentenv_protocol.website_browser.recording import BrowserRecordingManifest, BrowserRecordings

RED, BLUE, GREEN, YELLOW = (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)
ROOT = "agentenv-browser-recordings/r1"


def _grant() -> dict:
    return WriteNamespaceGrant(
        root_path=ROOT, expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        max_objects=1000, max_object_bytes=1 << 22, max_total_bytes=1 << 26,
        write=HttpPostPolicyGrant(kind="http-post-policy", url="https://objects.example.test/upload",
                                  fields={"policy": "p"}, path_field="key", file_field="file"),
    ).model_dump(mode="json")


def _jpeg(color: tuple[int, int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(out, "JPEG")
    return out.getvalue()


class Uploads:
    """Every upload, by grant root and path, in place of the object store."""

    def __init__(self) -> None:
        self.by_path: dict[str, bytes] = {}

    def uploader(self, grant):
        uploads = self

        class Uploader:
            async def upload(self, path, source):
                data = source if isinstance(source, bytes) else source.read_bytes()
                uploads.by_path[f"{grant.root_path}/{path}"] = data
        return Uploader()

    def colors(self, folder: str) -> list[str]:
        """The colour of every frame of the video under ``folder``, in order, by its strongest channel."""
        names = {RED: "red", BLUE: "blue", GREEN: "green", YELLOW: "yellow", (255, 255, 255): "white"}
        found = []
        for path in sorted(p for p in self.by_path if p.startswith(f"{folder}/video/") and p.endswith(".mp4")):
            with av.open(io.BytesIO(self.by_path[path])) as container:
                for frame in container.decode(video=0):
                    pixel = frame.to_image().getpixel((frame.width // 2, frame.height // 2))
                    found.append(names[min(names, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, pixel)))])
        return found


class Browser:
    """A browser's pipe, as recordings ask it: each tab attached on its own session, which a screencast paints in the
    tab's colour unless it's ``silent``, as a tab behind another is, hidden, and a capture answers in it; every other
    tab is visible, as each in front of its window is, and the one ``focused`` names has focus."""

    def __init__(self) -> None:
        self.listeners, self.colors, self.casting, self.silent, self.still = [], {}, set(), set(), set()
        self.focused, self.contexts, self.calls = None, 0, []

    def listen(self, listener) -> None:
        self.listeners.append(listener)

    def emit(self, method: str, params: dict, session=None) -> None:
        for listener in list(self.listeners):
            listener({"method": method, "params": params, **({"sessionId": session} if session else {})})

    def open(self, tab: str, color, context=None, opener=None) -> None:
        self.colors[tab] = color
        info = {"targetId": tab, "type": "page", "url": f"https://{tab.lower()}.test/",
                **({"browserContextId": context} if context else {}), **({"openerId": opener} if opener else {})}
        self.emit("Target.targetCreated", {"targetInfo": info})
        self.emit("Target.attachedToTarget", {"sessionId": f"s-{tab}", "targetInfo": info})

    def close(self, tab: str) -> None:
        self.emit("Target.detachedFromTarget", {"sessionId": f"s-{tab}"})
        self.emit("Target.targetDestroyed", {"targetId": tab})

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        self.calls.append((method, session_id))
        tab = session_id[2:] if session_id else None
        if method == "Target.createBrowserContext":
            self.contexts += 1
            return {"browserContextId": f"ctx-{self.contexts}"}
        if method == "Page.startScreencast":
            self.casting.add(session_id)
        elif method == "Page.stopScreencast":
            self.casting.discard(session_id)
        elif method == "Page.captureScreenshot":
            return {"data": base64.b64encode(_jpeg(self.colors[tab])).decode()}
        elif method == "Runtime.evaluate":
            shown = "hidden" if tab in self.silent else "visible"
            if params.get("expression") == "document.visibilityState":
                return {"result": {"value": shown}}
            return {"result": {"value": [shown, tab == self.focused]}}
        return {}

    async def paint(self) -> None:
        """Every screencast not silent painted, every 50 ms, until cancelled."""
        while True:
            for session in list(self.casting):
                if session[2:] not in self.silent | self.still and session[2:] in self.colors:
                    self.emit("Page.screencastFrame", {"data": base64.b64encode(_jpeg(self.colors[session[2:]]))
                                                       .decode(), "sessionId": 1, "metadata": {}}, session)
            await asyncio.sleep(0.05)


@pytest_asyncio.fixture
async def browser(monkeypatch, tmp_path):
    """A browser with its control, log, contexts, network log and recordings, its screencasts painting; and the
    uploads its recordings make."""
    uploads = Uploads()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(browser_recording, "NamespaceUploader", uploads.uploader)
    monkeypatch.setattr(computer_recording, "NamespaceUploader", uploads.uploader)
    monkeypatch.setattr(browser_recording, "SETTLE_S", 0.05)
    monkeypatch.setattr(browser_recording, "LOG_WAIT_S", 0.3)
    pipe = Browser()
    control = BrowserControl(pipe, "agent")
    control.log = log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    contexts = Contexts(pipe, control, home=lambda: tmp_path)
    network = NetworkLog(tmp_path / "network.jsonl")
    recordings = BrowserRecordings(log, control, contexts, network, (64, 48))
    painting = asyncio.ensure_future(pipe.paint())
    yield pipe, control, contexts, network, recordings, uploads
    painting.cancel()


def _acted(log: BrowserLog, tab, via: str = "cdp", method: str = "Input.dispatchMouseEvent", acts: bool = True,
           **fields) -> int:
    begun = log.begin({"via": via, "who": "ada", "method": method, "acts": acts, "tab": tab, **fields})
    log.finish(begun, ok=True)
    return begun["seq"]


def _network_lines(path, *keys: str, after: int = 0) -> None:
    with path.open("a") as file:
        for n, key in enumerate(keys, start=after + 1):
            file.write(json.dumps({"seq": 2 * n - 1, "at": "2026-11-13T10:00:00.000Z", "open": {
                "id": key, "kind": "exchange", "started": "2026-11-13T10:00:00.000Z",
                "request": {"method": "GET", "url": f"http://{key}.localhost/", "http_version": "HTTP/1.1",
                            "headers": []}}}) + "\n")
            file.write(json.dumps({"seq": 2 * n, "at": "2026-11-13T10:00:00.100Z", "end": {
                "id": key, "ended": "2026-11-13T10:00:00.100Z", "answered": "2026-11-13T10:00:00.050Z",
                "response": {"status": 200, "reason": "OK", "http_version": "HTTP/1.1", "headers": []}}}) + "\n")


@pytest.mark.asyncio
async def test_two_leases_recorded_at_once_each_film_only_their_own_tabs(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    buyer, seller = await contexts.lease("buyer"), await contexts.lease("seller")
    pipe.open("T0", GREEN)
    pipe.open("T1", RED, buyer.id)
    pipe.open("T2", BLUE, seller.id)
    pipe.focused = "T0"
    _network_lines(network.path, "start-tab")
    a = await recordings.start("run:a", _grant(), context="buyer", fps=10)
    b = await recordings.start("run:b", _grant(), context="seller", fps=10)
    await asyncio.sleep(0.4)
    _acted(recordings.log, "T2")
    _acted(recordings.log, "T0")
    await asyncio.sleep(0.4)
    first = BrowserRecordingManifest.model_validate(await recordings.end(a["recording_id"]))
    second = BrowserRecordingManifest.model_validate(await recordings.end(b["recording_id"]))

    assert set(uploads.colors(f"{ROOT}/{first.recording_id}")) == {"red"}
    assert set(uploads.colors(f"{ROOT}/{second.recording_id}")) == {"blue"}
    assert (first.context, second.context) == ("buyer", "seller") and a["format"].startswith("video/mp4")
    assert [(seen.tab, seen.via) for seen in first.video.followed] == [("T1", "screencast")]
    assert all(segment.uploaded and segment.path.startswith("video/") for segment in first.video.segments)
    assert first.ended_by == "end" and first.network.scope == "browser" and first.network.entries == 1


@pytest.mark.asyncio
async def test_the_video_follows_an_agents_tab_a_popup_it_opens_and_goes_back_when_that_closes(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    pipe.open("T0", GREEN)
    pipe.open("T3", YELLOW)
    pipe.focused = "T0"
    started = await recordings.start("run:a", _grant(), network=False)
    await asyncio.sleep(0.3)
    _acted(recordings.log, "T3")
    await asyncio.sleep(0.3)
    pipe.open("T4", RED, opener="T3")
    await asyncio.sleep(0.3)
    pipe.close("T4")
    await asyncio.sleep(0.3)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert [seen.tab for seen in manifest.video.followed] == ["T0", "T3", "T4", "T3"]
    colors = uploads.colors(f"{ROOT}/{manifest.recording_id}")
    assert colors[0] == "green" and "yellow" in colors and "red" in colors and colors[-1] == "yellow"


@pytest.mark.asyncio
async def test_a_clients_setup_on_every_page_it_attaches_to_doesnt_move_the_video_and_its_input_does(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    pipe.open("T0", GREEN)
    pipe.open("T3", YELLOW)
    pipe.focused = "T0"
    started = await recordings.start("run:a", _grant(), network=False)
    for method in ("Page.addScriptToEvaluateOnNewDocument", "Runtime.addBinding", "Emulation.setFocusEmulationEnabled",
                   "Network.setExtraHTTPHeaders", "Runtime.evaluate"):
        _acted(recordings.log, "T3", method=method)
    await asyncio.sleep(0.3)
    set_up = [seen["tab"] for seen in recordings.every()[0].stream.followed]
    _acted(recordings.log, "T3", method="Input.dispatchKeyEvent")
    await asyncio.sleep(0.3)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert set_up == ["T0"] and [seen.tab for seen in manifest.video.followed] == ["T0", "T3"]


@pytest.mark.asyncio
async def test_a_tab_that_sends_no_screencast_frame_is_captured_until_it_does(browser, monkeypatch):
    pipe, control, contexts, network, recordings, uploads = browser
    monkeypatch.setattr(browser_stream, "SILENT_S", 0.2)
    monkeypatch.setattr(browser_stream, "FALLBACK_FPS", 20)
    pipe.open("T1", RED)
    pipe.focused, pipe.silent = "T1", {"T1"}
    started = await recordings.start("run:a", _grant(), network=False)
    await asyncio.sleep(0.6)
    pipe.silent = set()
    await asyncio.sleep(0.4)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert [seen.via for seen in manifest.video.followed] == ["screencast", "screenshots", "screencast"]
    assert ("Page.captureScreenshot", "s-T1") in pipe.calls
    assert set(uploads.colors(f"{ROOT}/{manifest.recording_id}")[3:]) == {"red"}


@pytest.mark.asyncio
async def test_a_visible_tab_that_paints_nothing_is_a_still_page_its_last_frame_shows_uncaptured(browser,
                                                                                              monkeypatch):
    pipe, control, contexts, network, recordings, uploads = browser
    monkeypatch.setattr(browser_stream, "SILENT_S", 0.2)
    monkeypatch.setattr(browser_stream, "FALLBACK_FPS", 20)
    pipe.open("T1", RED)
    pipe.focused, pipe.still = "T1", {"T1"}
    started = await recordings.start("run:a", _grant(), network=False)
    await asyncio.sleep(0.8)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert [seen.via for seen in manifest.video.followed] == ["screencast"]
    assert pipe.calls.count(("Page.captureScreenshot", "s-T1")) == 1
    assert set(uploads.colors(f"{ROOT}/{manifest.recording_id}")) == {"red"}


@pytest.mark.asyncio
async def test_a_context_with_no_tab_is_blank_until_one_opens_there(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    buyer = await contexts.lease("buyer")
    started = await recordings.start("run:a", _grant(), context="buyer", network=False)
    await asyncio.sleep(0.3)
    pipe.open("T1", RED, buyer.id)
    await asyncio.sleep(0.4)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert [(seen.tab, seen.via) for seen in manifest.video.followed] == [(None, "blank"), ("T1", "screencast")]
    colors = uploads.colors(f"{ROOT}/{manifest.recording_id}")
    assert colors[0] == "white" and colors[-1] == "red"


@pytest.mark.asyncio
async def test_its_log_has_its_contexts_lines_and_none_of_anothers(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    buyer, seller = await contexts.lease("buyer"), await contexts.lease("seller")
    pipe.open("T1", RED, buyer.id)
    pipe.open("T2", BLUE, seller.id)
    started = await recordings.start("run:a", _grant(), context="buyer", network=False)
    mine = [_acted(recordings.log, "T1"), _acted(recordings.log, None, method="Target.getTargets", acts=False,
                                                 context="buyer")]
    _acted(recordings.log, "T2")
    _acted(recordings.log, None, method="Target.getTargets", acts=False, context="seller")
    _acted(recordings.log, None, via="agentenv", method="cookies/set")
    pipe.open("T5", YELLOW, buyer.id)
    pipe.open("T6", GREEN, seller.id)
    reset = recordings.log.note("reset", {})
    await asyncio.sleep(0.2)
    manifest = await recordings.end(started["recording_id"])
    lines = [json.loads(line) for line in uploads.by_path[
        f"{ROOT}/{manifest['recording_id']}/log.jsonl"].decode().splitlines()]

    actions = [line["seq"] for line in lines if "action" in line]
    opened = [line["tab"]["id"] for line in lines if "tab" in line]
    assert actions == mine and opened == ["T5"] and reset in [line["seq"] for line in lines]


@pytest.mark.asyncio
async def test_its_network_log_is_the_browsers_whole_log_as_har_marking_what_it_wrote_meanwhile(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    await contexts.lease("buyer")
    pipe.open("T0", GREEN)
    _network_lines(network.path, "start-tab")
    started = await recordings.start("run:a", _grant())
    _network_lines(network.path, "during", after=1)
    await contexts.lease("later")
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))
    document = json.loads(uploads.by_path[f"{ROOT}/{manifest.recording_id}/network.har"])

    assert [entry["request"]["url"] for entry in document["log"]["entries"]] == [
        "http://start-tab.localhost/", "http://during.localhost/"]
    assert manifest.network.model_dump() == {"file": "network.har", "scope": "browser", "uploaded": True,
                                             "entries": 2, "during": [3, 4], "contexts_open": ["buyer", "later"],
                                             "error": None}


@pytest.mark.asyncio
async def test_a_start_is_refused_with_no_network_log_an_unleased_context_a_bad_fps_or_no_encoder(browser,
                                                                                                    monkeypatch):
    pipe, control, contexts, network, recordings, uploads = browser
    alone = BrowserRecordings(recordings.log, control, contexts, None)

    with pytest.raises(Unsupported, match="no network log"):
        await alone.start("run:a", _grant())
    with pytest.raises(TargetUnavailable, match="'nope'"):
        await recordings.start("run:b", _grant(), context="nope")
    with pytest.raises(InvalidArgument, match="fps is an integer from 1 to 15"):
        await recordings.start("run:c", _grant(), fps=60)
    monkeypatch.setattr(browser_recording, "Video", None)
    with pytest.raises(Unsupported, match="av and Pillow"):
        await recordings.start("run:d", _grant())


@pytest.mark.asyncio
async def test_a_context_an_agent_makes_over_its_contexts_address_is_recorded_with_it_and_ends_with_its_context(
        browser):
    pipe, control, contexts, network, recordings, uploads = browser
    seller = await contexts.lease("seller")
    pipe.open("T2", BLUE, seller.id)
    started = await recordings.start("run:a", _grant(), context="seller", network=False)
    made = contexts.made_by_agent("bob", "ctx-bob", scope=seller.id)
    contexts.made_by_agent("ada", "ctx-ada", scope=None)
    pipe.open("T7", YELLOW, "ctx-bob")
    await asyncio.sleep(0.4)
    contexts.disposed("ctx-bob")
    await asyncio.sleep(0.4)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))
    child = manifest.children[0]
    folder = f"{ROOT}/{manifest.recording_id}/contexts/{made.label}"
    kept = BrowserRecordingManifest.model_validate_json(uploads.by_path[f"{folder}/manifest.json"])

    assert [c.context for c in manifest.children] == [made.label] and child.folder == f"contexts/{made.label}/"
    assert (child.made_by, child.ended_by, child.missing) == ("bob", "context_ended", None)
    assert kept.context == made.label and set(uploads.colors(folder)) <= {"white", "yellow"}
    assert "yellow" in uploads.colors(folder) and set(uploads.colors(f"{ROOT}/{manifest.recording_id}")) == {"blue"}


@pytest.mark.asyncio
async def test_a_lease_released_ends_its_recording_saying_so_and_an_end_after_answers_the_same(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    buyer = await contexts.lease("buyer")
    pipe.open("T1", RED, buyer.id)
    started = await recordings.start("run:a", _grant(), context="buyer", frames=True, network=False)
    await asyncio.sleep(0.2)
    await contexts.release("buyer")
    await asyncio.sleep(0.3)
    ended = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"], _grant()))

    assert ended.ended_by == "context_ended" and ended.missing_frames == 0 and ended.video.segments
    assert ended.model_dump(mode="json") == await recordings.end(started["recording_id"])


@pytest.mark.asyncio
async def test_a_frame_follows_each_settled_burst_of_an_agents_actions_on_its_contexts_tab_keyed_to_its_entries(
        browser):
    pipe, control, contexts, network, recordings, uploads = browser
    buyer = await contexts.lease("buyer")
    pipe.open("T1", RED, buyer.id)
    pipe.open("T2", BLUE, buyer.id)
    pipe.open("T9", GREEN)
    started = await recordings.start("run:a", _grant(), frames=True, context="buyer", network=False)
    typed = [_acted(recordings.log, "T1", method="Input.insertText") for _ in range(3)]
    _acted(recordings.log, "T1", method="DOM.getDocument", acts=False)
    _acted(recordings.log, "T9")
    await asyncio.sleep(0.2)
    clicked = _acted(recordings.log, "T2", via="mcp", method="browser_click")
    await asyncio.sleep(0.2)
    manifest = BrowserRecordingManifest.model_validate(await recordings.end(started["recording_id"]))

    assert [(f.kind, f.after, f.tab) for f in manifest.frames][1:] == [
        ("action", typed, "T1"), ("action", [clicked], "T2"), ("last", [], "T2")]
    assert all(f.uploaded and f.content_type == "image/jpeg" for f in manifest.frames)
    assert uploads.by_path[f"{ROOT}/{manifest.recording_id}/frames/0001.jpg"] == _jpeg(RED)
    assert manifest.missing_frames == 0 and manifest.frames_on


@pytest.mark.asyncio
async def test_without_frames_a_recording_takes_none_and_still_uploads_its_log(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    pipe.open("T1", RED)
    started = await recordings.start("run:a", _grant(), network=False)
    _acted(recordings.log, "T1")
    await asyncio.sleep(0.2)
    manifest = await recordings.end(started["recording_id"])

    assert manifest["frames"] == [] and manifest["log_uploaded"] and manifest["frames_on"] is False


@pytest.mark.asyncio
async def test_a_start_again_joins_its_recording_an_end_again_answers_the_same_and_an_unknown_one_is_refused(
        browser):
    pipe, control, contexts, network, recordings, uploads = browser
    pipe.open("T1", RED)
    first = await recordings.start("run:a", _grant(), network=False)
    again = await recordings.start("run:a", _grant(), network=False)
    ended = await recordings.end(first["recording_id"])

    assert again == first and await recordings.end(first["recording_id"]) == ended
    with pytest.raises(InvalidArgument, match="no recording nope"):
        await recordings.end("nope")
    with pytest.raises(InvalidArgument, match="reason is one of"):
        await recordings.end(first["recording_id"], reason="later")
    with pytest.raises(InvalidArgument, match="write namespace grant"):
        await recordings.start("run:b", {"root_path": "x"}, network=False)


@pytest.mark.asyncio
async def test_every_open_recording_ends_as_its_env_closes(browser):
    pipe, control, contexts, network, recordings, uploads = browser
    pipe.open("T1", RED)
    await recordings.start("run:a", _grant(), network=False)
    await recordings.start("run:b", _grant(), frames=True, network=False)

    ended = await recordings.end_all("reset")

    assert [m["ended_by"] for m in ended] == ["reset", "reset"] and await recordings.end_all() == []


@pytest.mark.asyncio
async def test_one_screencast_serves_everything_following_its_tab_and_stops_once_nothing_does():
    pipe = Browser()
    control = BrowserControl(pipe, "agent")
    pipe.open("T1", RED)
    heard_a, heard_b = [], []

    session = await control.screencast("T1", heard_a.append)
    await control.screencast("T1", heard_b.append)
    pipe.emit("Page.screencastFrame", {"data": base64.b64encode(b"x").decode(), "sessionId": 7}, "s-T1")
    await asyncio.sleep(0)
    await control.unscreencast(session, heard_a.append)
    stopped_early = ("Page.stopScreencast", "s-T1") in pipe.calls
    await control.unscreencast(session, heard_b.append)

    assert heard_a == heard_b == [b"x"] and ("Page.screencastFrameAck", "s-T1") in pipe.calls
    assert [call for call in pipe.calls if call[0] == "Page.startScreencast"] == [("Page.startScreencast", "s-T1")]
    assert not stopped_early and ("Page.stopScreencast", "s-T1") in pipe.calls
    assert await control.screencast("nope", heard_a.append) is None


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

    data, _, tab, _ = await control.frame("T1", load_wait_s=0)

    (_, _, first), (_, _, second) = pipe.captured
    assert (base64.b64decode(data), tab) == (b"behind", "T1") and second - first >= NUDGE_S


@pytest.mark.asyncio
async def test_a_tab_that_closes_while_its_page_loads_is_framed_as_the_active_tab():
    pipe, control = _two_tabs()
    pipe.emit("Page.frameNavigated", {"frame": {"id": "T2", "url": "https://a.test/pop", "loaderId": "L2"}}, "s-T2")
    asyncio.get_running_loop().call_later(0.1, pipe.emit, "Target.detachedFromTarget", {"sessionId": "s-T2"})

    _, _, tab, _ = await control.frame("T2", load_wait_s=5)

    assert tab != "T2" and [session for session, _, _ in pipe.captured] == [f"s-{tab}"]
