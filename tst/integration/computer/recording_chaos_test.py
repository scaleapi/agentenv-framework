"""computer-recording/v1 on the scaffold's desktop when things go wrong: the X server killed mid-recording, segments
rolling over, the disk filling up, clicks while it records at 30 fps, and the driver restarted. Each test runs the
scaffold's image in a container of its own, so it can kill, fill and restart it, and decodes every segment it reads
back. Needs a Docker daemon."""

import asyncio
import json
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import av
import pytest

from agentenv_protocol.client import ExtensionCallError
from agentenv_protocol.computer import COMPUTER_READY, COMPUTER_RECORDING, Recording, RecordingManifest
from agentenv_protocol.computer.conformance.client import Driver
from agentenv_protocol.computer.recording import DISK_CHECK_S, RECORDINGS_DIR
from tst.integration.computer.conftest import docker
from tst.integration.computer.scaffold_test import _scaffold, _when

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

PORT = 18765
RECORDINGS = f"/home/user/{RECORDINGS_DIR}"
SEGMENT_S = Recording().max_segment_s
FPS = 10


@pytest.fixture(scope="module")
def image(tmp_path_factory, base):
    tag = f"agentenv-test-recording-desk:{uuid.uuid4().hex[:8]}"
    scaffold = _scaffold(tmp_path_factory.mktemp("desk") / "scaffold", base)
    docker("build", "-q", "-t", tag, str(scaffold / "envs" / "desktop"))
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.mark.asyncio
async def test_a_recording_outlives_its_x_server_with_a_gap_and_a_new_segment(image, tmp_path):
    async with _desk(image) as (name, driver):
        started = await _start(driver)
        await asyncio.sleep(3)
        docker("exec", name, "pkill", "-9", "Xvfb")
        await asyncio.sleep(6)
        manifest = await _end(driver, started)
        decoded = _decoded(name, manifest, tmp_path)
    assert manifest.ended_by == "end" and len(manifest.segments) >= 2
    assert max(_gaps(manifest)) >= 0.5 and decoded == [s.frames for s in manifest.segments]


@pytest.mark.asyncio
async def test_segments_roll_over_at_max_segment_s_with_no_frame_lost_between_them(image, tmp_path):
    async with _desk(image) as (name, driver):
        started = await _start(driver)
        await asyncio.sleep(2 * SEGMENT_S + 5)
        manifest = await _end(driver, started)
        decoded = _decoded(name, manifest, tmp_path)
    spans = [(_when(s.ended_at) - _when(s.started_at)).total_seconds() for s in manifest.segments]
    gaps = _gaps(manifest)
    assert len(spans) >= 3 and all(SEGMENT_S - 2 / FPS <= span <= SEGMENT_S for span in spans[:2]), spans
    assert all(gap < 2 / FPS for gap in gaps), gaps
    assert decoded == [s.frames for s in manifest.segments]


@pytest.mark.asyncio
async def test_a_full_disk_ends_a_recording_as_limit_and_what_it_recorded_plays(image, tmp_path):
    tmpfs = f"type=tmpfs,destination={RECORDINGS},tmpfs-size={600 << 20},tmpfs-mode=1777"
    async with _desk(image, "--mount", tmpfs) as (name, driver):
        started = await _start(driver)
        await asyncio.sleep(2)
        docker("exec", name, "dd", "if=/dev/zero", f"of={RECORDINGS}/fill", "bs=1M", "count=150")
        await asyncio.sleep(2 * DISK_CHECK_S)
        manifest = await _end(driver, started)
        decoded = _decoded(name, manifest, tmp_path)
    assert manifest.ended_by == "limit" and decoded and decoded == [s.frames for s in manifest.segments]


@pytest.mark.asyncio
async def test_every_click_lands_while_the_display_records_at_30_fps(image):
    async with _desk(image) as (_name, driver):
        layout = await _layout(driver)
        (gx, gy), cell = layout["grid"][:2], layout["cell"]
        started = await _start(driver, fps=30)
        async with driver.session() as session:
            await session.call_tool("computer_screenshot", {})
            for i in range(64):
                x, y = gx + (i % 8) * cell + cell // 2, gy + (i // 8) * cell + cell // 2
                clicked = await session.call_tool("computer_click", {"x": x, "y": y})
                assert not clicked.isError, clicked.content[0].text
        await asyncio.sleep(1)
        cells = json.loads((await driver.run("cat", "fixture/state.json"))["stdout"])["cells"]
        manifest = await _end(driver, started)
    span = (_when(manifest.segments[-1].ended_at) - _when(manifest.segments[0].started_at)).total_seconds()
    assert len(cells) == 64 and sum(s.frames for s in manifest.segments) / span > 25


@pytest.mark.asyncio
async def test_a_restarted_driver_forgets_its_recording_and_keeps_the_segments_it_closed(image):
    async with _desk(image) as (name, driver):
        started = await _start(driver)
        await asyncio.sleep(SEGMENT_S + 2)
        docker("restart", "-t", "2", name)
        driver = await _ready(name)
        with pytest.raises(ExtensionCallError) as refused:
            await _end(driver, started)
        kept = (await driver.run("ls", f"{RECORDINGS}/{started['recording_id']}"))["stdout"].split()
    assert refused.value.code == "invalid_argument" and "0000.mp4" in kept


@asynccontextmanager
async def _desk(image: str, *docker_args: str):
    """The desktop in a container of its own, and a client for its driver once it is ready."""
    name = f"agentenv-test-desk-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, "-p", f"127.0.0.1::{PORT}", *docker_args,
            "-e", 'AGENTENV_COMPUTER_SETTINGS={"data_paths": ["fixture"]}', image)
    try:
        yield name, await _ready(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def _ready(name: str) -> Driver:
    """A client for the container's driver once its computer is ready; a restart may move its port."""
    for _ in range(240):
        port = subprocess.run(["docker", "port", name, str(PORT)], capture_output=True, text=True).stdout
        if port:
            try:
                driver = await Driver.connect(f"http://{port.splitlines()[0]}", 30)
                if (await driver.invoke(COMPUTER_READY, "status"))["ready"]:
                    return driver
            except Exception:  # not serving yet
                pass
        await asyncio.sleep(0.25)
    raise TimeoutError(f"{name}'s computer never became ready")


async def _layout(driver: Driver) -> dict:
    for _ in range(120):
        out = await driver.run("cat", "fixture/layout.json")
        if out["exit_code"] == 0 and out["stdout"].strip():
            return json.loads(out["stdout"])
        await asyncio.sleep(0.25)
    raise TimeoutError("the Fixture Desk never wrote its layout")


async def _start(driver: Driver, fps: int = FPS) -> dict:
    return await driver.invoke(COMPUTER_RECORDING, "start", {"request_id": uuid.uuid4().hex, "fps": fps})


async def _end(driver: Driver, started: dict) -> RecordingManifest:
    return RecordingManifest.model_validate(
        await driver.invoke(COMPUTER_RECORDING, "end", {"recording_id": started["recording_id"]}))


def _decoded(name: str, manifest: RecordingManifest, tmp: Path) -> list[int]:
    """How many frames each segment decodes to, read back out of the container."""
    counts = []
    for segment in manifest.segments:
        local = tmp / segment.path.replace("/", "-")
        local.write_bytes(docker("exec", name, "cat", f"{RECORDINGS}/{segment.path}").stdout)
        with av.open(str(local)) as video:
            counts.append(sum(1 for _ in video.decode(video=0)))
    return counts


def _gaps(manifest: RecordingManifest) -> list[float]:
    segments = manifest.segments
    return [(_when(b.started_at) - _when(a.ended_at)).total_seconds() for a, b in zip(segments, segments[1:])]
