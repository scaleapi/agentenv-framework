"""The shipped desktop example through ``agent-env run``: its computer env, deployed on local Docker, passes the
conformance probe at full; a file loaded into it is read by code run inside it and checked by a command there; a
scripted drive of the conformance fixture on it is recorded on the clock its screenshots carry; and an agent given the
computer's skill files a loaded file in a new folder through the computer_* tools under a recording. Needs Docker and
the local registry; the agent's task, a model endpoint too."""

import json
from datetime import datetime
from pathlib import Path

import av
import pytest
from click.testing import CliRunner
from PIL import Image

from agent_env.artifact import Artifact
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.store.routing import namespace_routing
from tst.integration.computer.conftest import run_context
from tst.util.capabilities import skip_without_model_endpoint

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

RECORDING = Path(__file__).parent / "recording"


def test_the_desktop_examples_computer_passes_the_probe_at_full(state):
    ran = _run("desktop", "probe")

    assert ran.exit_code == 0, ran.output
    assert "passed (probe: 1)" in ran.output, ran.output


def test_a_file_loaded_into_the_desktop_is_read_by_code_inside_it_and_checked_there(state):
    ran = _run("desktop", "files")

    assert ran.exit_code == 0, ran.output
    assert "passed (tests: 1)" in ran.output, ran.output
    context = run_context(ran.output)
    (loaded,) = context["metadata"]["loaded_file_artifact_universes"]
    assert (loaded["destination_path"], loaded["files"]) == ("Documents", {"figures.csv": "Documents/figures.csv"})
    assert context["metadata"]["script_results"]["inside"] == {"header": "quarter,revenue", "card": "desktop"}


def test_a_scripted_drive_is_recorded_on_the_clock_its_screenshots_carry(state):
    ran = _run(str(RECORDING), "recording")

    assert ran.exit_code == 0, ran.output
    assert "passed (drive: 1)" in ran.output, ran.output
    context, manifest, frames = _recorded(ran.output, "desk", state)
    entry = context["metadata"]["computer_recordings"]["desk"]
    assert (entry["state"], entry["ended_by"], entry["missing_segments"]) == ("ended", "end", 0)
    assert manifest["source"] == "stream" and manifest["cursor"] is True and frames
    assert len(frames) == sum(s["frames"] for s in manifest["segments"])
    drive = json.loads(context["metadata"]["verifications"]["drive"]["results"][0]["description"])
    interval = 1 / manifest["fps"]
    for shot in drive["shots"]:
        at = _when(shot["captured_at"]).timestamp()
        nearest = min(frames, key=lambda frame: abs(frame[0] - at))
        assert abs(nearest[0] - at) <= interval, (shot["step"], nearest[0] - at)
        diff = sum(abs(a - b) for a, b in zip(_thumb(nearest[1]), bytes.fromhex(shot["thumb"]))) / (64 * 40)
        assert diff < 6, (shot["step"], diff)
    x, y, w, h = drive["clicked_cell"]
    shots = drive["shots"]
    n = next(i for i, s in enumerate(shots) if s["args"].get("x") == x + w // 2)
    corner = lambda image: image.getpixel((x + 8, y + 8))  # noqa: E731  clear of the pointer drawn at the click
    before = _when(shots[n - 1]["captured_at"]).timestamp()
    _, unclicked = min(frames, key=lambda frame: abs(frame[0] - before))
    changed = next(t for t, image in frames if t > before and corner(image) != corner(unclicked))
    # The cell changes after the screenshot before its click, by the one after it, both on the computer's clock.
    assert changed <= _when(shots[n]["captured_at"]).timestamp() + interval


@skip_without_model_endpoint()
def test_an_agent_given_the_computers_skill_files_a_loaded_file_in_a_new_folder_under_a_recording(state):
    ran = _run("desktop", "agent")

    assert ran.exit_code == 0, ran.output
    assert "passed (check: 1)" in ran.output, ran.output
    context, manifest, frames = _recorded(ran.output, "agent", state)
    entry = context["metadata"]["computer_recordings"]["agent"]
    assert (entry["state"], entry["missing_segments"]) == ("ended", 0) and frames
    assert "## Shortcuts" in context["metadata"]["env_skills"]["default-agent"]["desktop"]["desktop"]


def _run(bundle: str, task: str):
    return CliRunner().invoke(cli, ["run", bundle, "--task", task])


def _recorded(output: str, name: str, tmp: Path) -> tuple[dict, dict, list]:
    """The run's context, the manifest of its recording ``name``, and every frame of the video with its time on the
    computer's clock."""
    context = run_context(output)
    entry = context["metadata"]["computer_recordings"][name]
    with namespace_routing():
        universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
        objects = get_config().get_object_store()
        files = {name: objects.get(Artifact.get(ref).object_url) for name, ref in universe.file_artifact_ids.items()}
    manifest = json.loads(files["manifest.json"])
    frames = []
    for segment in manifest["segments"]:
        path = tmp / segment["path"].replace("/", "-")
        path.write_bytes(files[segment["path"].replace("/", "-")])
        with av.open(str(path)) as video:
            start = _when(segment["started_at"])
            frames += [(start.timestamp() + frame.time, frame.to_image()) for frame in video.decode(video=0)]
    return context, manifest, frames


def _when(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _thumb(image: Image.Image) -> bytes:
    return image.convert("L").resize((64, 40)).tobytes()
