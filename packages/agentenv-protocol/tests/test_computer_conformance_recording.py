"""The probe's recording checks against a served driver that records: they pass, and they fail on a manifest whose
hash or size isn't the file's, on a card that declares a limit the driver doesn't keep, and on an MP4 that isn't one;
every recording the probe made is ended and removed."""
from __future__ import annotations

import asyncio
import struct
from datetime import datetime, timezone

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")
pytest.importorskip("av")

from probe_fake import FixtureDriver, served  # noqa: E402

from agentenv_protocol.computer import RecordedFrame, Recording, capture, record  # noqa: E402
from agentenv_protocol.computer import recording as recording_module  # noqa: E402
from agentenv_protocol.computer.conformance import checks, run_probe  # noqa: E402
from agentenv_protocol.computer.conformance.client import Driver  # noqa: E402
from agentenv_protocol.computer.conformance.mp4 import read_mp4  # noqa: E402

NAMES = ("a recording's segments match their files, with no gap between them",
         "a repeated request_id joins the first recording", "a recording past max_concurrent is refused")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(checks, "RECORD_S", 0.8)
    return tmp_path


class Recorder(FixtureDriver):
    recording = Recording(max_concurrent=2)


async def _recording_checks(driver) -> dict[str, tuple[str, str]]:
    with served(driver) as url:
        connected = await Driver.connect(url, 10)
        async with connected.session() as session:
            probe = checks.Probe(connected, session, 10)
            await probe._recording()
    return {r.name: (r.status, r.detail) for r in probe.results}


@pytest.mark.asyncio
async def test_a_driver_that_records_passes_the_recording_checks_at_full_and_leaves_nothing(home):
    with served(Recorder()) as url:
        report = await run_probe(url, "full")
    statuses = {c.name: c.status for c in report.checks}
    assert report.passed and all(statuses[name] == "passed" for name in NAMES)
    assert report.extensions["urn:agentenv:computer-recording/v1"] == "verified"
    assert not (home / ".agentenv-recordings").exists()


@pytest.mark.asyncio
async def test_a_manifest_whose_hash_or_size_isnt_the_files_fails(home, monkeypatch):
    real = recording_module._digest
    monkeypatch.setattr(recording_module, "_digest", lambda path: (real(path)[0], "0" * 64))
    wrong_hash = await _recording_checks(Recorder())
    monkeypatch.setattr(recording_module, "_digest", lambda path: (real(path)[0] + 1, real(path)[1]))
    wrong_size = await _recording_checks(Recorder())
    assert wrong_hash[NAMES[0]][0] == "failed" and "sha256 " in wrong_hash[NAMES[0]][1]
    assert wrong_size[NAMES[0]][0] == "failed" and "bytes" in wrong_size[NAMES[0]][1]


@pytest.mark.asyncio
async def test_a_limit_the_driver_doesnt_keep_fails_and_its_recordings_are_still_ended(home, monkeypatch):
    advertised = recording_module.recording_params
    monkeypatch.setattr("agentenv_protocol.computer.card.recording_params",
                        lambda features, streams: {**advertised(features, streams), "max_concurrent": 1})
    driver = Recorder()
    found = await _recording_checks(driver)
    assert found[NAMES[2]][0] == "failed" and "recording 2 of at most 1 started" in found[NAMES[2]][1]
    assert found[NAMES[0]][0] == found[NAMES[1]][0] == "passed"
    assert not driver._computer.recordings.running and not (home / ".agentenv-recordings").exists()


@pytest.mark.asyncio
async def test_a_driver_that_turns_recording_off_gets_no_recording_checks(home):
    class Off(FixtureDriver):
        recording = None

    assert await _recording_checks(Off()) == {}


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def test_the_mp4_reader_refuses_what_isnt_an_mp4_with_a_video_track():
    with pytest.raises(ValueError, match="not an MP4"):
        read_mp4(_box(b"free", b""))
    hdlr = _box(b"hdlr", b"\0" * 8 + b"soun" + b"\0" * 12)
    audio = _box(b"ftyp", b"isom") + _box(b"moov", _box(b"trak", _box(b"mdia", hdlr)))
    with pytest.raises(ValueError, match="no video track"):
        read_mp4(audio)
    with pytest.raises(ValueError, match="runs past its container"):
        read_mp4(_box(b"ftyp", b"isom") + struct.pack(">I4s", 400, b"moov"))
    vide = _box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    no_tkhd = _box(b"ftyp", b"isom") + _box(b"moov", _box(b"trak", _box(b"mdia", vide)))
    with pytest.raises(ValueError, match="no tkhd box"):
        read_mp4(no_tkhd)


def _mp4(*, large: bool, version: int) -> bytes:
    tkhd = b"\0" * (88 if version else 76) + struct.pack(">II", 320 << 16, 200 << 16)
    tkhd = bytes([version]) + tkhd[1:]
    mdhd = (bytes([1]) + b"\0" * 19 + struct.pack(">IQ", 1000, 2500)) if version else (
        b"\0" * 12 + struct.pack(">II", 1000, 2500))
    stts = b"\0" * 4 + struct.pack(">III", 1, 25, 100)
    vide = _box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    mdia = _box(b"mdia", vide + _box(b"mdhd", mdhd) + _box(b"minf", _box(b"stbl", _box(b"stts", stts))))
    moov = _box(b"moov", _box(b"trak", _box(b"tkhd", tkhd) + mdia))
    if large:
        moov = struct.pack(">I4sQ", 1, b"moov", 16 + len(moov) - 8) + moov[8:]
    return _box(b"ftyp", b"isom") + moov + struct.pack(">I4s", 0, b"mdat") + b"frames"


@pytest.mark.parametrize("large, version", [(False, 0), (True, 1)])
def test_the_mp4_reader_takes_large_boxes_a_box_to_the_end_and_version_1_headers(large, version):
    assert read_mp4(_mp4(large=large, version=version)) == {"width": 320, "height": 200, "frames": 25,
                                                            "duration_s": 2.5}


class Gappy(Recorder):
    """Its captures fail once, a little into a recording."""
    calls = 0

    @capture
    async def grab(self):
        Gappy.calls += 1
        if Gappy.calls == 3:
            raise RuntimeError("the X server went away")
        return await super().grab()


@pytest.mark.asyncio
async def test_each_way_a_recording_can_fall_short_fails_its_check(home, monkeypatch):
    monkeypatch.setattr(checks, "RECORD_S", 2.2)
    gap = await _recording_checks(Gappy())
    monkeypatch.setattr(checks, "RECORD_S", 0.8)
    monkeypatch.setattr(recording_module, "MIN_FREE_BYTES", 1 << 62)
    limited = await _recording_checks(Recorder())
    monkeypatch.setattr(recording_module, "MIN_FREE_BYTES", 0)
    real_digest = checks.READ_DIGEST
    monkeypatch.setattr(checks, "READ_DIGEST", "import sys; sys.exit(3)")
    unreadable = await _recording_checks(Recorder())
    monkeypatch.setattr(checks, "READ_DIGEST", real_digest)
    monkeypatch.setattr(checks, "READ_CHUNK", "import sys; sys.exit(3)")
    truncated = await _recording_checks(Recorder())
    monkeypatch.undo()
    monkeypatch.setattr(checks, "RECORD_S", 0.8)
    real_end = recording_module.Recordings.end

    async def altered(change):
        async def end(self, recording_id, upload, reason):
            manifest = await real_end(self, recording_id, upload, reason)
            change(manifest)
            return manifest

        monkeypatch.setattr(recording_module.Recordings, "end", end)
        return await _recording_checks(Recorder())

    miscount = await altered(lambda m: m["segments"][0].update(frames=m["segments"][0]["frames"] + 1))
    unmeasured = await altered(lambda m: m["segments"][0].pop("sha256"))
    malformed = await altered(lambda m: m.pop("ended_by"))
    assert "s between segments 0 and 1" in gap[NAMES[0]][1]
    assert "it ended by 'limit'" in limited[NAMES[0]][1]
    assert "couldn't be read back" in unreadable[NAMES[0]][1]
    assert "stopped reading back at byte 0" in truncated[NAMES[0]][1]
    assert "the manifest says 1280x800" in miscount[NAMES[0]][1]
    assert "segments [0] lack their times" in unmeasured[NAMES[0]][1]
    assert "isn't a computer-recording/v1 manifest" in malformed[NAMES[0]][1]


@pytest.mark.asyncio
async def test_a_card_with_no_max_concurrent_fails_the_limit_check(home, monkeypatch):
    advertised = recording_module.recording_params
    monkeypatch.setattr("agentenv_protocol.computer.card.recording_params",
                        lambda features, streams: {k: v for k, v in advertised(features, streams).items()
                                                   if k != "max_concurrent"})
    found = await _recording_checks(Recorder())
    assert found[NAMES[2]] == ("failed", "the card declares no max_concurrent")


class Stalled(Recorder):
    """A stream that hands over one frame and then nothing."""

    @record
    async def frames(self, hints):
        yield RecordedFrame(self.screen.image.copy(), datetime.now(timezone.utc))
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_stream_that_stalls_after_its_first_frame_fails(home):
    found = await _recording_checks(Stalled())
    assert found[NAMES[0]][0] == "failed" and "1 frames over 0.00 s of a 0.8 s recording" in found[NAMES[0]][1]


@pytest.mark.asyncio
async def test_a_segment_bigger_than_a_chunk_is_read_back_in_pieces(home, monkeypatch):
    monkeypatch.setattr(checks, "CHUNK_BYTES", 1024)
    found = await _recording_checks(Recorder())
    assert found[NAMES[0]] == ("passed", "")
