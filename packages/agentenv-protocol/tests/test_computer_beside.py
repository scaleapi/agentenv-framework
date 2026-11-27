"""A driver beside an Android device: the base class's Android defaults go through adb under /sdcard, write atomically,
keep output under its cap, share one upload grant with the recordings kept in the driver's home, and fail rather than
look empty; its skill names the device's user and home; and the probe reads a recording back through data/get, with no
python3."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver, http_client, rpc  # noqa: E402

from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI  # noqa: E402
from agentenv_protocol.computer import _adb  # noqa: E402
from agentenv_protocol.computer.conformance.checks import Probe  # noqa: E402
from agentenv_protocol.computer.errors import DeliveryFailed, DeviceUnavailable, InvalidArgument  # noqa: E402
from agentenv_protocol.computer.dataplane import RECORDINGS_DIR  # noqa: E402
from agentenv_protocol.computer.manifest import ManifestSegment  # noqa: E402
from agentenv_protocol.computer.platform import Platform  # noqa: E402

ANDROID = Platform(os="android", arch="amd64", display="none")
FAKE_ADB = Path(__file__).with_name("fake_adb.py")


class BesideDriver(FakeDriver):
    placement = "beside"
    default_data_paths = ("Download", "DCIM")

    def __init__(self) -> None:
        super().__init__(platform=ANDROID)


@pytest.fixture
def device(tmp_path, monkeypatch):
    (tmp_path / "sdcard").mkdir()
    adb = tmp_path / "fake-adb"
    adb.write_text(f"#!{sys.executable}\n{FAKE_ADB.read_text()}")
    adb.chmod(0o755)
    monkeypatch.setenv("AGENTENV_ADB", str(adb))
    monkeypatch.setenv("FAKE_DEVICE", str(tmp_path / "sdcard"))
    monkeypatch.setenv("ANDROID_SERIAL", "emulator-5554")
    monkeypatch.setenv("ADB_SERVER_SOCKET", "tcp:10.0.0.2:5037")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "sdcard"


def _calls(device) -> list[dict]:
    path = device.parent / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _part(path: str, data: bytes) -> dict:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode()},
            "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}


@pytest.mark.asyncio
async def test_commands_and_files_go_through_adb_under_sdcard_and_recordings_stay_in_the_drivers_home(device):
    kept = device.parent / "home" / RECORDINGS_DIR / "rec-1"
    kept.mkdir(parents=True)
    (kept / "0.mp4").write_bytes(b"mp4")
    driver = BesideDriver()
    async with http_client(driver) as http:
        await http.post("/agentenv", json=rpc("data/add", {"parts": [_part("Download/f.csv", b"1,2")]}))
        got = (await http.post("/agentenv", json=rpc("data/get", {"paths": ["Download", f"{RECORDINGS_DIR}/rec-1"]}))
               ).json()["result"]["parts"]
        card = (await http.get("/.well-known/agent-env.json")).json()
    ran = await driver._computer.call("run_in_session", ["test", "-f", "f.csv"], "Download", None, 10)

    assert {p["metadata"][DATA_OBJECTS_EXTENSION_URI]["path"]: base64.b64decode(p["file"]["bytes"]) for p in got} == {
        "Download/f.csv": b"1,2", f"{RECORDINGS_DIR}/rec-1/0.mp4": b"mp4"}
    assert ran.exit_code == 0
    computer = {e["uri"]: e for e in card["capabilities"]["extensions"]}["urn:agentenv:computer/v1"]["params"]
    assert (computer["placement"], computer["default_data_paths"]) == ("beside", ["Download", "DCIM"])
    calls = _calls(device)
    pushed = [call["args"][2] for call in calls if call["args"][0] == "push"]
    shell = [call["args"][1] for call in calls if call["args"][0] == "shell"]
    assert (device / "Download" / "f.csv").read_bytes() == b"1,2"
    assert not list((device / "Download").glob(".agentenv-*"))
    assert len(pushed) == 1 and pushed[0].startswith("/sdcard/Download/.agentenv-")
    assert "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/Download/f.csv" in shell
    assert {(call["serial"], call["server"]) for call in calls} == {("emulator-5554", "tcp:10.0.0.2:5037")}


@pytest.mark.asyncio
async def test_the_skill_of_a_driver_beside_a_device_names_the_devices_user_and_home():
    async with http_client(BesideDriver()) as http:
        skill_md = (await http.get("/agentenv/ext/skill/get", params={"name": "computer-desktop"})).json()["skill_md"]
    assert "- You act as `shell`, whose home is `/sdcard`." in skill_md


class _ProbedDriver:
    """What the probe sees of a beside driver: its card, and a data/get that answers one kept segment. It has no
    computer-exec, so a probe that ran python3 on the device would fail."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.asked: list = []

    def params(self, uri: str) -> dict:
        return {"placement": "beside", "platform": ANDROID.as_dict()}

    async def rpc(self, method: str, params: dict | None = None) -> dict:
        self.asked.append((method, params))
        path = params["paths"][0]
        return {"result": {"parts": [{"kind": "file", "file": {"bytes": base64.b64encode(self.data).decode()},
                                      "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}]}}

    async def run(self, *argv: str, timeout_s: float = 30) -> dict:
        raise AssertionError(f"the probe ran {argv} on a device with no python3")


@pytest.mark.asyncio
async def test_the_probe_reads_a_beside_drivers_recording_back_through_data_get_without_python3():
    data = b"segment bytes"
    driver = _ProbedDriver(data)
    probe = Probe(driver, session=None, ready_timeout_s=1)
    segment = ManifestSegment(index=0, path="rec-1/0.mp4", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                              uploaded=False)
    assert await probe._read_back(segment) == data
    assert driver.asked == [("data/get", {"paths": [f"{RECORDINGS_DIR}/rec-1/0.mp4"]})]
    wrong = segment.model_copy(update={"bytes": 99})
    assert "the manifest says 99" in await probe._read_back(wrong)


@pytest.mark.asyncio
async def test_a_commands_output_is_kept_only_up_to_the_cap(device):
    result = await _adb.run_in_session(["sh", "-c", "head -c 100000 /dev/zero"], None, None, 10, output_bytes=100)
    assert (len(result.stdout), result.truncated, result.exit_code) == (100, True, 0)


@pytest.mark.asyncio
async def test_device_files_and_recordings_upload_under_one_grants_limits(device):
    (device / "Download").mkdir()
    (device / "Download" / "f.csv").write_bytes(b"1,2")
    kept = device.parent / "home" / RECORDINGS_DIR / "rec-1"
    kept.mkdir(parents=True)
    (kept / "0.mp4").write_bytes(b"mp4")
    grant = {"root_path": "runs/1", "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
             "max_objects": 5, "max_object_bytes": 100, "max_total_bytes": 500,
             "write": {"kind": "http-post-policy", "url": "https://store.test/", "fields": {},
                       "path_field": "key", "file_field": "file"}}
    with patch("agentenv_protocol.computer.dataplane.NamespaceUploader") as uploader:
        uploader.return_value.upload = AsyncMock()
        async with http_client(BesideDriver()) as http:
            await http.post("/agentenv", json=rpc("data/get", {"write_namespace": grant,
                                                                "paths": ["Download", f"{RECORDINGS_DIR}/rec-1"]}))
    assert uploader.call_count == 1
    assert [call.args[0] for call in uploader.return_value.upload.await_args_list] == [
        "Download/f.csv", f"{RECORDINGS_DIR}/rec-1/0.mp4"]


@pytest.mark.asyncio
async def test_failures_fail_loudly_and_leave_the_device_as_it_was(device, monkeypatch, tmp_path):
    (device / "Download").mkdir()
    (device / "Download" / "f.csv").write_bytes(b"old")
    monkeypatch.setenv("FAKE_PUSH_FAILS", "1")
    with pytest.raises(DeliveryFailed, match="connection reset"):
        await _adb.write_files({"Download/f.csv": b"new contents"})
    assert [p.name for p in (device / "Download").iterdir()] == ["f.csv"]
    assert (device / "Download" / "f.csv").read_bytes() == b"old"
    monkeypatch.delenv("FAKE_PUSH_FAILS")
    with pytest.raises(InvalidArgument, match="'Download' is a folder on the device"):
        await _adb.write_files({"Download": b"x"})
    monkeypatch.setenv("FAKE_OFFLINE", "1")
    with pytest.raises(DeliveryFailed, match="device offline"):
        await _adb.read_files(["Download"], tmp_path)
    with pytest.raises(DeviceUnavailable, match="device offline"):
        await _adb.run_in_session(["true"], None, None, 10, output_bytes=100)


@pytest.mark.asyncio
async def test_a_program_or_folder_that_isnt_there_is_refused_and_a_command_failing_is_a_result(device, monkeypatch):
    run = _adb.run_in_session
    with pytest.raises(InvalidArgument, match="not found on the device"):
        await run(["agentenv-probe-no-such-program"], None, None, 10, output_bytes=100)
    with pytest.raises(InvalidArgument, match="no such folder"):
        await run(["true"], "Nowhere", None, 10, output_bytes=100)
    failed = await run(["sh", "-c", 'printf "$B"; exit 127'], None, {"A;touch x": "y", "B": "it's"}, 10,
                       output_bytes=100)
    assert (failed.exit_code, failed.stdout) == (127, b"it's") and not (device / "x").exists()
    monkeypatch.setenv("FAKE_DAEMON_STARTS", "1")
    ran = await run(["sh", "-c", "echo oops >&2"], None, None, 10, output_bytes=100)
    assert (ran.exit_code, ran.stderr) == (0, b"oops\n")
    (device / "bin").mkdir()
    (device / "bin" / "agentenv-hello").write_text("#!/bin/sh\necho hello\n")
    (device / "bin" / "agentenv-hello").chmod(0o755)
    found = await run(["agentenv-hello"], None, {"PATH": "/sdcard/bin:/usr/bin:/bin"}, 10, output_bytes=100)
    assert found.stdout == b"hello\n"


@pytest.mark.asyncio
async def test_a_cancelled_read_stops_adb(device, monkeypatch, tmp_path):
    (device / "Download").mkdir()
    monkeypatch.setenv("FAKE_PULL_HANGS", "1")
    reading = asyncio.ensure_future(_adb.read_files(["Download"], tmp_path))
    while not (device.parent / "adb.pid").exists():
        await asyncio.sleep(0.05)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    with pytest.raises(ProcessLookupError):
        os.kill(int((device.parent / "adb.pid").read_text()), 0)
