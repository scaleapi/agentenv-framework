"""The reference driver on a desktop of its own before agent-env can compose a computer: the scaffold, an image under
this folder that lays it on the ubuntu-24.04 base, deployed by ``agent-env run`` as an MCP server on the server
provider, passes the conformance probe at full three runs in a row, records a scripted drive of its desk that lines up
with the drive's screenshots, and an agent drives its desktop through the computer_* tools under a recording. Needs a
Docker daemon and the local registry; the agent's task, a model endpoint too."""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import av
import pytest
from click.testing import CliRunner
from PIL import Image

import agent_env
import agentenv_protocol
from agent_env.artifact import Artifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, get_config, reset_config
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from agent_env.task.store import get_task_instance_store
from tst.util.capabilities import skip_without_model_endpoint

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

SCAFFOLD = Path(__file__).parent / "scaffold"
DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "reference" / "driver.py"
FIXTURE = Path(agentenv_protocol.__file__).parent / "computer" / "conformance" / "fixture.py"
PROTOCOL = Path(agentenv_protocol.__file__).parents[2]
PROTOCOL_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _scaffold(root: Path, base: str) -> Path:
    """The scaffold bundle on the base image ``base``, with the reference driver, the probe's fixture and this
    checkout's protocol beside its Dockerfile."""
    shutil.copytree(SCAFFOLD, root, ignore=shutil.ignore_patterns("__pycache__"))
    desktop = root / "envs" / "desktop"
    dockerfile = desktop / "Dockerfile"
    text = dockerfile.read_text()
    assert "\nARG BASE\n" in text, "the scaffold's Dockerfile names its base by ARG BASE, which _scaffold sets"
    dockerfile.write_text(text.replace("\nARG BASE\n", f"\nARG BASE={base}\n"))
    shutil.copy(DRIVER, desktop / "driver.py")
    shutil.copy(FIXTURE, desktop / "fixture.py")
    shutil.copytree(PROTOCOL / "src", desktop / "protocol" / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in PROTOCOL_FILES:
        shutil.copy(PROTOCOL / name, desktop / "protocol" / name)
    return root


def _run(root: Path, task: str):
    return CliRunner().invoke(cli, ["run", str(root), "--task", task])


def _recorded(output: str, name: str, tmp: Path) -> tuple[dict, dict, list]:
    """The run's context, the manifest of its recording ``name``, and every frame of the video with its time on the
    computer's clock."""
    instance_id = re.search(r"instance (@local/\S+)", output).group(1)
    with namespace_routing():
        context = get_task_instance_store().get(instance_id).context
        entry = context["metadata"]["computer_recordings"][name]
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


def test_the_reference_driver_passes_the_probe_at_full(state, base):
    ran = _run(_scaffold(state / "scaffold", base), "probe")

    assert ran.exit_code == 0, ran.output
    assert "passed (probe: 1)" in ran.output, ran.output


def test_a_scripted_drive_is_recorded_on_the_clock_its_screenshots_carry(state, base):
    ran = _run(_scaffold(state / "scaffold", base), "recording")

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
    changed = next(t for t, image in frames if corner(image) != corner(frames[0][1]))
    # The cell changes after the screenshot before its click and by the one after it, both on the computer's clock.
    assert _when(shots[n - 1]["captured_at"]).timestamp() < changed <= _when(shots[n]["captured_at"]).timestamp() + interval


@skip_without_model_endpoint()
def test_an_agent_drives_the_reference_drivers_desktop_under_a_recording(state, base):
    ran = _run(_scaffold(state / "scaffold", base), "agent")

    assert ran.exit_code == 0, ran.output
    assert "passed (check: 1)" in ran.output, ran.output
    context, manifest, frames = _recorded(ran.output, "agent", state)
    assert context["metadata"]["computer_recordings"]["agent"]["state"] == "ended" and frames
