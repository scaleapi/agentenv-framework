"""computer-recording/v1 ``annotate``: a recording started with it draws each input where it landed, in the colour
of the role that sent it, on the frames captured after it; a call to the driver's own or forwarded tool that acts
shows as a chip; a plain recording draws nothing."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")
av = pytest.importorskip("av")

from computer_fake import FakeDriver, MinimalDriver, http_client, mcp_session  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol import EnvironmentTool, ForwardedTool, tool  # noqa: E402
from agentenv_protocol.computer import (  # noqa: E402
    COMPUTER_RECORDING,
    DeviceUnavailable,
    RecordedSegment,
    Recording,
    record,
)
from agentenv_protocol.computer.annotate import (  # noqa: E402
    COLOURS,
    FADE_S,
    KEEP_S,
    STYLE,
    UNIT_EDGE,
    Annotations,
    draw,
    input_marks,
    tool_mark,
)
from agentenv_protocol.computer.conformance import checks  # noqa: E402
from agentenv_protocol.tool_context import ROLE_META_KEY  # noqa: E402

START, END = "/agentenv/ext/computer-recording/start", "/agentenv/ext/computer-recording/end"
T0 = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
CLICK = (400, 300)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class Recorded(FakeDriver):
    recording = Recording(max_segment_s=60)


class Pointerless(MinimalDriver):
    """A driver with no pointer group, so its recordings, polled from captures, can't draw the pointer themselves."""
    recording = Recording(max_segment_s=60)


class Segmented(FakeDriver):
    """A native recorder that hands over a finished segment, which the base class keeps as it came."""
    recording = Recording(max_segment_s=60)

    @record
    async def segments(self, hints):
        path = hints.directory / "0000.mp4"
        start = datetime.now(timezone.utc)
        out = av.open(str(path), "w", format="mp4")
        stream = out.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 100, "yuv420p"
        for i in range(3):
            frame = av.VideoFrame.from_image(PILImage.new("RGB", (160, 100))).reformat(format="yuv420p")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()
        yield RecordedSegment(path, start, start + timedelta(seconds=0.2), 160, 100, 3)
        await asyncio.sleep(3600)


class Templated(Recorded):
    """A driver with tools of its own named after its environment: one that only reads, and one that acts whose name
    ends as the first's does."""

    @tool(name="{environment_name}_inspect", annotations={"readOnlyHint": True})
    async def inspect(self) -> str:
        return "read"

    @tool(name="{environment_name}_foo_inspect")
    async def poke(self) -> str:
        return "poked"


class Forwarding(Recorded):
    """A driver serving two tools of another server's: one that acts, one its annotations say only reads."""

    def forwarded_tools(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        return [ForwardedTool(EnvironmentTool(name="click", description="Click an element.", inputSchema=schema),
                              self.answer),
                ForwardedTool(EnvironmentTool(name="get_window_state", description="Read a window.",
                                              inputSchema=schema), self.answer, {"readOnlyHint": True})]

    async def answer(self, **arguments):
        return CallToolResult(content=[TextContent(type="text", text="done")])


async def _clicked(driver, start: dict, meta: dict | None = None) -> tuple[datetime, dict, list]:
    """A recording started with ``start`` around one click at CLICK: when the click was sent, the manifest, and each
    frame with its capture time."""
    async with http_client(driver) as http, mcp_session(driver) as session:
        started = (await http.post(START, json={"request_id": "r", "fps": 10, **start})).json()
        await asyncio.sleep(0.5)
        shot = await session.call_tool("computer_screenshot", {})
        sent = datetime.now(timezone.utc)
        await session.call_tool("computer_click", {"x": CLICK[0], "y": CLICK[1],
                                                   "frame_id": shot.structuredContent["frame_id"]}, meta=meta)
        await asyncio.sleep(0.5)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"]})).json()
    return sent, manifest, _timed_frames(manifest)


def _timed_frames(manifest: dict) -> list[tuple[datetime, PILImage.Image]]:
    frames = []
    for segment in manifest["segments"]:
        started = datetime.fromisoformat(segment["started_at"].replace("Z", "+00:00"))
        path = f"{os.environ['HOME']}/.agentenv-recordings/{segment['path']}"
        with av.open(path) as container:
            decoded = list(container.decode(container.streams.video[0]))
        for frame in decoded:
            frames.append((started + timedelta(seconds=frame.time - decoded[0].time), frame.to_image()))
    return frames


def _marked(image: PILImage.Image, point: tuple[int, int], colour: tuple[int, int, int], reach: int = 40) -> int:
    """How many pixels within ``reach`` of ``point`` are near ``colour``, as an encoded frame keeps it."""
    x, y = point
    near = 0
    for px in range(max(0, x - reach), min(image.width, x + reach)):
        for py in range(max(0, y - reach), min(image.height, y + reach)):
            if sum(abs(a - b) for a, b in zip(image.getpixel((px, py)), colour)) < 90:
                near += 1
    return near


def _split(frames: list, sent: datetime) -> tuple[list, list]:
    """The frames captured before the click was sent, and those while its mark shows."""
    before = [image for at, image in frames if at < sent - timedelta(milliseconds=50)]
    after = [image for at, image in frames if sent + timedelta(milliseconds=150) <= at <= sent + timedelta(seconds=0.5)]
    assert before and after, f"{len(frames)} frames, {len(before)} before the click and {len(after)} after it"
    return before, after


@pytest.mark.asyncio
async def test_an_annotated_recording_draws_a_click_where_it_landed_on_the_frames_after_it():
    sent, manifest, frames = await _clicked(Recorded(), {"annotate": {}})

    before, after = _split(frames, sent)
    assert manifest["annotate"] == {}
    assert not any(_marked(image, CLICK, COLOURS[0]) for image in before)
    assert all(_marked(image, CLICK, COLOURS[0]) > 20 for image in after)


@pytest.mark.asyncio
async def test_a_plain_recording_draws_nothing_and_its_manifest_names_no_annotate():
    sent, manifest, frames = await _clicked(Recorded(), {})

    _before, after = _split(frames, sent)
    assert "annotate" not in manifest
    assert not any(_marked(image, CLICK, COLOURS[0]) for image in after)


@pytest.mark.asyncio
async def test_each_role_draws_in_its_own_colour():
    sent, _manifest, frames = await _clicked(Recorded(), {"annotate": {}}, meta={ROLE_META_KEY: "user"})

    _before, after = _split(frames, sent)
    assert all(_marked(image, CLICK, COLOURS[1]) > 20 and not _marked(image, CLICK, COLOURS[0]) for image in after)


@pytest.mark.asyncio
async def test_a_driver_tool_that_acts_shows_as_a_chip_and_one_that_only_reads_or_the_sdks_own_does_not():
    driver = Forwarding()
    async with http_client(driver) as http, mcp_session(driver) as session:
        started = (await http.post(START, json={"request_id": "r", "fps": 10, "annotate": {}})).json()
        recording = driver._computer.recordings.running[started["recording_id"]]
        await session.call_tool("click", {"text": "Save"})
        await session.call_tool("get_window_state", {"text": "Notes"})
        await session.call_tool("computer_screenshot", {})
        labels = [mark.label for mark in recording.annotations.marks]
        await http.post(END, json={"recording_id": started["recording_id"]})

    assert labels == ['click "Save"']


@pytest.mark.asyncio
async def test_start_refuses_an_annotate_that_isnt_an_object_or_names_an_option_it_doesnt_take():
    async with http_client(Recorded()) as http:
        option = await http.post(START, json={"request_id": "a", "annotate": {"events": True}})
        scalar = await http.post(START, json={"request_id": "b", "annotate": True})

    assert option.status_code == scalar.status_code == 400
    assert option.json()["error"]["code"] == scalar.json()["error"]["code"] == "invalid_argument"
    assert "takes no options yet, got ['events']" in option.json()["error"]["message"]


@pytest.mark.asyncio
async def test_the_card_lists_annotate_and_its_options_with_the_recording_params():
    async with http_client(Recorded()) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()

    [extension] = [e for e in card["capabilities"]["extensions"] if e["uri"] == COMPUTER_RECORDING]
    assert extension["params"]["annotate"] == {"options": []}


def test_each_input_is_marked_by_what_it_did_and_who_did_it():
    def labels(operation, args, role=None, **kw):
        return [(m.kind, m.label) for m in input_marks(operation, args, role=role, at=T0, size=(1000, 500), **kw)]

    assert labels("click", (500, 250, "left", 2, ["Shift"])) == [("click", ""), ("chip", "Shift + click")]
    assert labels("type_text", ("Q4  165\n",)) == [("chip", 'type "Q4  165[Enter]"')]
    assert labels("press", (["Control", "s"],), repeat=3) == [("chip", "key Ctrl+S x3")]
    assert labels("key_down", (["Shift"],), hold_ms=500) == [("chip", "hold Shift 500 ms")]
    assert labels("pointer_down", ("left",)) == [("chip", "press left")]
    assert labels("pointer_down", ("left",), point=(10, 20)) == [("down", "")]
    assert labels("navigate", ("https://example.com",), role="user") == [("chip", "user: open https://example.com")]
    assert labels("tabs", ("list", None, None)) == []
    [click] = input_marks("click", (500, 250, "right", 1, []), role=None, at=T0, size=(1000, 500))
    assert (click.points, click.button, click.role) == (((0.5, 0.5),), "right", "default")
    [drag] = input_marks("drag", ([(0, 0), (1000, 500)], "left", 400, []), role=None, at=T0, size=(1000, 500))
    assert drag.moving_s == 0.4 and drag.until - drag.at == timedelta(seconds=1.2)


def test_a_tools_chip_shows_its_main_argument():
    def label(arguments):
        return tool_mark("hotkey", arguments, role=None, at=T0).label

    assert label({"keys": ["cmd", "s"]}) == "hotkey cmd+S"
    assert label({"text": "x" * 100}) == 'hotkey "' + "x" * 59 + '…"'
    assert label({"window_id": 4}) == "hotkey"
    assert tool_mark("click", {"element": 7}, role="executor", at=T0).label == "executor: click 7"


def test_a_mark_shows_from_its_input_until_it_ends_and_is_then_forgotten():
    annotations = Annotations()
    annotations.add(input_marks("click", (10, 10, "left", 1, []), role=None, at=T0, size=(100, 100)))

    assert annotations.showing(T0 - timedelta(milliseconds=10)) == []
    assert len(annotations.showing(T0 + timedelta(milliseconds=500))) == 1
    assert annotations.showing(T0 + timedelta(seconds=1)) == [] and annotations.marks == []


def test_a_mark_lands_at_its_point_on_a_frame_of_any_size():
    """A frame scaled down by max_edge, or captured at a HiDPI scale, keeps the point where the input landed."""
    annotations = Annotations()
    annotations.add(input_marks("click", (500, 250, "left", 1, []), role=None, at=T0, size=(1000, 500)))
    at = T0 + timedelta(milliseconds=400)

    for size in ((1000, 500), (400, 200), (2000, 1000)):
        image = draw(PILImage.new("RGB", size, (255, 255, 255)), annotations.showing(at))
        cx, cy = size[0] // 2, size[1] // 2
        reach = round(STYLE.dot * max(size) / UNIT_EDGE) + 3  # the click's hollow dot, around its point
        assert any(image.getpixel((cx + dx, cy + dy)) == COLOURS[0]
                   for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)), size
        assert image.getpixel((5, 5)) == (255, 255, 255)


def test_a_mark_no_frame_caught_while_it_lasted_shows_fully_on_the_first_frame_after_it_once():
    """At a low fps, or after a slow capture, the next frame may come after a click's mark is over."""
    annotations = Annotations()
    [click] = input_marks("click", (10, 10, "left", 1, []), role=None, at=T0, size=(100, 100))
    annotations.add([click])

    [(mark, _colour, age)] = annotations.showing(T0 + timedelta(seconds=3))
    span = (click.until - click.at).total_seconds()
    assert mark is click and age == pytest.approx(span - FADE_S)
    assert annotations.showing(T0 + timedelta(seconds=4)) == [] and annotations.marks == []


def test_marks_no_frame_draws_are_kept_only_so_long():
    annotations = Annotations()
    annotations.add(input_marks("click", (10, 10, "left", 1, []), role=None, at=T0, size=(100, 100)))
    later = T0 + timedelta(seconds=KEEP_S + 1)
    annotations.add(input_marks("click", (20, 20, "left", 1, []), role=None, at=later, size=(100, 100)))

    assert [mark.at for mark in annotations.marks] == [later]


@pytest.mark.asyncio
async def test_an_input_goes_ahead_when_its_mark_cant_be_made():
    """Reading the display for a mark fails: the scroll is still delivered, unmarked."""
    driver = Recorded()
    async with http_client(driver) as http, mcp_session(driver) as session:
        started = (await http.post(START, json={"request_id": "r", "fps": 10, "annotate": {}})).json()
        recording = driver._computer.recordings.running[started["recording_id"]]
        await session.call_tool("computer_screenshot", {})
        driver.fail["display"] = DeviceUnavailable("the display went away")
        scrolled = await session.call_tool("computer_scroll", {"direction": "down"})
        marks = list(recording.annotations.marks)
        await http.post(END, json={"recording_id": started["recording_id"]})

    assert not scrolled.isError and driver.screen.log[-1][0] == "scroll" and marks == []


@pytest.mark.asyncio
async def test_a_recording_of_a_native_recorders_finished_segments_names_no_annotate():
    driver = Segmented()
    async with http_client(driver) as http:
        started = (await http.post(START, json={"request_id": "r", "annotate": {}})).json()
        await asyncio.sleep(0.3)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"]})).json()

    assert manifest["segments"] and "annotate" not in manifest


def test_chips_past_the_most_a_frame_shows_wait_for_a_later_frame_rather_than_go_unseen():
    annotations = Annotations()
    for n in range(6):
        annotations.add([tool_mark(f"tool{n}", {}, role=None, at=T0 + timedelta(milliseconds=n))])

    first = [mark.label for mark, _colour, _age in annotations.showing(T0 + timedelta(milliseconds=100))]
    later = [mark.label for mark, _colour, _age in annotations.showing(T0 + timedelta(seconds=3))]

    assert first == ["tool2", "tool3", "tool4", "tool5"] and later == ["tool0", "tool1"]
    assert annotations.showing(T0 + timedelta(seconds=4)) == [] and annotations.marks == []


@pytest.mark.asyncio
async def test_the_probe_finds_annotation_not_applicable_to_a_native_recorders_finished_segments():
    wire = {"recording_id": "rec-1", "ended_by": "end", "segments": []}
    with pytest.raises(checks.NotApplicable, match="hands over finished segments"):
        await checks.Probe._annotated_click(None, wire, (100, 100), (10, 10), (20, 20), (255, 255, 255))


@pytest.mark.asyncio
async def test_a_tool_named_after_its_environment_is_told_read_only_by_its_own_annotations():
    driver = Templated()
    async with http_client(driver) as http, mcp_session(driver) as session:
        names = [t.name for t in (await session.list_tools()).tools if t.name.endswith("_inspect")]
        started = (await http.post(START, json={"request_id": "r", "fps": 10, "annotate": {}})).json()
        recording = driver._computer.recordings.running[started["recording_id"]]
        for name in names:
            await session.call_tool(name, {})
        labels = [mark.label for mark in recording.annotations.marks]
        await http.post(END, json={"recording_id": started["recording_id"]})

    [poke] = [name for name in names if name.endswith("_foo_inspect")]
    assert len(names) == 2 and labels == [poke]


def _pointed(image: PILImage.Image, tip: tuple[int, int], colour: tuple[int, int, int]) -> int:
    """How many pixels near ``colour`` the agent's pointer drawn at ``tip`` would cover: its arrow, down and right."""
    x, y = tip
    return sum(1 for px in range(x, min(image.width, x + 14)) for py in range(y + 2, min(image.height, y + 22))
               if sum(abs(a - b) for a, b in zip(image.getpixel((px, py)), colour)) < 90)


@pytest.mark.parametrize("driver, drawn", [(Pointerless, True), (Recorded, False)],
                         ids=["frames-without-a-pointer", "frames-with-the-pointer"])
@pytest.mark.asyncio
async def test_the_agents_pointer_is_drawn_where_its_click_left_it_only_where_the_frames_cant_show_the_pointer(
        driver, drawn):
    """Once the click's own mark is over, a frame still shows the pointer there: drawn in the agent's colour where
    the recording's frames leave the pointer out, and left to the frames' own where they draw it."""
    driver = driver()
    async with http_client(driver) as http, mcp_session(driver) as session:
        started = (await http.post(START, json={"request_id": "r", "fps": 10, "annotate": {}})).json()
        await asyncio.sleep(0.3)
        shot = await session.call_tool("computer_screenshot", {})
        sent = datetime.now(timezone.utc)
        await session.call_tool("computer_click", {"x": 200, "y": 150, "frame_id": shot.structuredContent["frame_id"]})
        await asyncio.sleep(1.6)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"]})).json()

    after = [image for at, image in _timed_frames(manifest)
             if sent + timedelta(seconds=1.0) <= at <= sent + timedelta(seconds=1.5)]
    assert after and all((_pointed(image, (200, 150), COLOURS[0]) > 10) is drawn for image in after)


def test_the_pointer_follows_the_last_input_before_each_frame_from_the_first_input_on():
    annotations = Annotations(pointer=True)
    annotations.add(input_marks("click", (10, 10, "left", 1, []), role=None, at=T0, size=(100, 100)))
    annotations.add(input_marks("pointer_move", (50, 50), role=None, at=T0 + timedelta(seconds=1), size=(100, 100)))

    def pointer(seconds):
        return [(mark.points[0], mark.points[-1]) for mark, _colour, _age in
                annotations.showing(T0 + timedelta(seconds=seconds)) if mark.kind == "pointer"]

    assert pointer(-1) == []
    assert pointer(0.5) == [((0.1, 0.1), (0.1, 0.1))]
    assert pointer(1.5) == [((0.1, 0.1), (0.5, 0.5))] and pointer(9) == [((0.1, 0.1), (0.5, 0.5))]
