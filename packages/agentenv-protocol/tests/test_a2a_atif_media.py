"""An agent that serves atif/1 spools its copies' media files to its disk, one file per sha256, with each image's size
read from its header, and uploads the ones the worker asks for under exact-object grants (``TRAJECTORY_V1.media``),
which is how the worker pulls a turn's media at its end. Readers take a part path only as media/<sha256>.<ext>, and
only with the bytes that name."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import struct
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient
from test_a2a_atif_serving import _atif, _ended, _send

from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    TaskObjectTrajectoryRequest,
    TaskTrajectoryRequest,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    TrajectoryState,
    Uploaded,
    a2a_agent,
    enable,
    extension,
)
from agentenv_protocol.a2a_agent import _media_spool
from agentenv_protocol.a2a_agent._media_spool import MediaSpool
from agentenv_protocol.a2a_agent._atif_copy import TurnStart
from agentenv_protocol.a2a_agent._trajectory_log import TaskTrajectories
from agentenv_protocol.atif import Adapter, AtifError, atif_adapter, check_media, errors_in, media_file, native, validate
from agentenv_protocol.atif._image_size import image_size
from agentenv_protocol.transfers import TransferError, WriteObject

NATIVE = "shots/1"


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    pixels = zlib.compress(b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height)))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", pixels) + chunk(b"IEND", b""))


SHOT = _png(4, 3)


@atif_adapter(native_format=NATIVE, version="1", input_includes_cache=False)
class Shots(Adapter):
    @native(type="shot")
    def shot(self, record: dict[str, Any], turn: Any) -> None:
        data = base64.b64decode(record["data"])
        media = turn.media(data, record["media_type"], **record.get("info", {}))
        turn.message(step=record["step"], content=[media])
        turn.step(key=record["step"], llm_call_count=1)


def _shooting(*, atif: bool = True) -> AgentEnvAgent:
    """An agent whose turn appends each base64 file its prompt's JSON lists as a native ``shot``."""
    trajectory = enable(TRAJECTORY_V1, live=True, native=NATIVE, **({"atif": Shots} if atif else {}))

    @a2a_agent(identity=AgentIdentity(name="shots", description="test", version="1"), extensions=(trajectory,))
    class Shooting(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format(NATIVE)
            for index, part in enumerate(request.parts[1:]):
                request.trajectory.append({"type": "shot", "step": f"s{index}", "data": part.bytes,
                                           "media_type": part.mime_type})
            return TaskResult.text("done")

    return Shooting()


def _file(data: bytes, media_type: str, name: str = "file") -> dict[str, Any]:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode(), "mimeType": media_type, "name": name}}


def _grant(path: str, media_type: str = "image/png") -> dict[str, Any]:
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {"media_type": media_type, "max_bytes": 10_000_000,
            "write": {"kind": "http-put", "url": f"https://objects.example.test/{path}", "expires_at": expires}}


def _path(data: bytes, extension: str) -> str:
    return f"media/{hashlib.sha256(data).hexdigest()}{extension}"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Where the media calls' grants upload: each object by its path."""
    stored: dict[str, bytes] = {}

    async def upload(target, source) -> Uploaded:
        data = Path(source).read_bytes()
        stored[target.write.url.removeprefix("https://objects.example.test/")] = data
        return Uploaded(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    monkeypatch.setattr(_media_spool, "upload", upload)
    return stored


def _spool(client: TestClient) -> MediaSpool:
    return client.app.state.agentenv_a2a.services.task_trajectories.media


def _media(client: TestClient, paths: dict[str, str], context_id: str = "ctx-1") -> Any:
    return client.post("/ext/trajectory/media", json={"context_id": context_id, "objects": {
        path: _grant(path, media_type) for path, media_type in paths.items()}})


def test_a_copys_images_are_spooled_by_sha256_with_their_size_and_pushed_once_asked(store: dict[str, bytes]) -> None:
    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png", "inbound.png")])
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
        spool = _spool(client)
        path = _path(SHOT, ".png")
        held = spool.held("ctx-1", path)
        assert held.read_bytes() == SHOT
        pushed = _media(client, {path: "image/png"})
        assert pushed.status_code == 200, pushed.text
        assert spool.held("ctx-1", path) is None and not held.exists()
    assert errors_in(validate(document)) == []
    [user] = [step for step in document["steps"] if step["source"] == "user"]
    image = next(part for part in user["message"] if isinstance(part, dict) and part.get("type") == "image")
    assert image["source"]["path"] == path
    assert user["extra"]["agentenv"]["media"][path] == {"sha256": hashlib.sha256(SHOT).hexdigest(),
                                                       "bytes": len(SHOT), "width": 4, "height": 3, "name": "inbound.png"}
    assert pushed.json() == {"objects": {path: {"size_bytes": len(SHOT), "sha256": hashlib.sha256(SHOT).hexdigest()}},
                             "missing": [], "failed": {}}
    assert store == {path: SHOT}


def test_a_media_calls_grants_leave_the_copys_media_paths_as_they_are(store: dict[str, bytes]) -> None:
    """A grant's URL may name the file it writes, ``media/<sha256>.<ext>``: the scrub takes a URL's credential from it,
    never the file's name, which every read of the copy keeps."""
    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")])
        _ended(client, task_id)
        assert _media(client, {_path(SHOT, ".png"): "image/png"}).status_code == 200
        document = _atif(client, task_id=task_id)["trajectory"]
    [user] = [step for step in document["steps"] if step["source"] == "user"]
    assert any(part.get("source", {}).get("path") == _path(SHOT, ".png") for part in user["message"]
               if isinstance(part, dict))


def test_the_same_bytes_added_twice_are_spooled_and_pushed_once(store: dict[str, bytes]) -> None:
    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png"), _file(SHOT, "image/png")])
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
        assert sorted(file.name for file in _spool(client).folder.iterdir()) == [_path(SHOT, ".png").removeprefix("media/")]
        pushed = _media(client, {_path(SHOT, ".png"): "image/png"})
    paths = [part["source"]["path"] for step in document["steps"] for part in step.get("message") or []
             if isinstance(part, dict) and part.get("type") == "image"]
    assert paths == [_path(SHOT, ".png")] * 4
    assert list(pushed.json()["objects"]) == [_path(SHOT, ".png")] and len(store) == 1


def test_a_path_the_context_doesnt_hold_and_a_failed_upload_dont_stop_the_rest(
        store: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    other = _png(2, 2)
    stored = _media_spool.upload

    async def upload(target, source) -> Uploaded:
        if target.write.url.endswith(_path(other, ".png")):
            raise TransferError("grant_expired", "The transfer grant expired.")
        return await stored(target, source)

    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png"), _file(other, "image/png")])
        _ended(client, task_id)
        monkeypatch.setattr(_media_spool, "upload", upload)
        never = _path(b"never added", ".png")
        pushed = _media(client, {_path(SHOT, ".png"): "image/png", _path(other, ".png"): "image/png",
                                 never: "image/png"}).json()
        assert _spool(client).held("ctx-1", _path(other, ".png")) is not None
        other_context = _media(client, {_path(SHOT, ".png"): "image/png"}, context_id="ctx-2").json()
    assert list(pushed["objects"]) == [_path(SHOT, ".png")] and pushed["missing"] == [never]
    assert pushed["failed"] == {_path(other, ".png"): "grant_expired: The transfer grant expired."}
    assert other_context == {"objects": {}, "missing": [_path(SHOT, ".png")], "failed": {}}


def test_an_agent_that_serves_its_own_trajectory_reads_still_has_the_sdks_media_and_cant_take_it_over(
        store: dict[str, bytes]) -> None:
    @a2a_agent(identity=AgentIdentity(name="own-reads", description="test", version="1"),
               extensions=(enable(TRAJECTORY_V1, native=NATIVE, atif=Shots),))
    class OwnReads(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format(NATIVE)
            return TaskResult.text("done")

        @extension(TRAJECTORY_V1.get)
        async def trajectory(self, request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest) -> dict[str, Any]:
            return {"trajectory": []}

    with TestClient(OwnReads().create_app()) as client:
        _ended(client, _send(client, "look", "m-1", extra_parts=[_file(SHOT, "image/png")]))
        assert list(_media(client, {_path(SHOT, ".png"): "image/png"}).json()["objects"]) == [_path(SHOT, ".png")]

    @a2a_agent(identity=AgentIdentity(name="takes-media", description="test", version="1"),
               extensions=(enable(TRAJECTORY_V1, live=True, native=NATIVE, atif=Shots),))
    class TakesMedia(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("done")

        @extension(TRAJECTORY_V1.media.media)
        async def media(self, request: Any) -> dict[str, Any]:
            return {"objects": {}}

    with pytest.raises(ValueError, match="served by the SDK alone"):
        TakesMedia().create_app()


def test_an_upload_that_outlives_its_copy_releases_only_the_hold_it_began_with(
        monkeypatch: pytest.MonkeyPatch) -> None:
    spool = MediaSpool()
    path = _path(SHOT, ".png")

    async def upload(target, source) -> Uploaded:
        spool.drop("ctx")
        spool.sink("ctx")(path, SHOT)
        return Uploaded(size_bytes=len(SHOT), sha256=hashlib.sha256(SHOT).hexdigest())

    monkeypatch.setattr(_media_spool, "upload", upload)
    spool.sink("ctx")(path, SHOT)
    pushed = asyncio.run(spool.push("ctx", {path: WriteObject.model_validate(_grant(path))}))
    assert list(pushed.objects) == [path]
    assert spool.held("ctx", path).read_bytes() == SHOT
    spool.close()


def test_a_spooled_file_that_no_longer_matches_its_name_never_reaches_the_store(store: dict[str, bytes]) -> None:
    with TestClient(_shooting().create_app()) as client:
        _ended(client, _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")]))
        _spool(client).held("ctx-1", _path(SHOT, ".png")).write_bytes(b"changed on disk")
        pushed = _media(client, {_path(SHOT, ".png"): "image/png"}).json()
    assert pushed["objects"] == {} and store == {}
    assert pushed["failed"][_path(SHOT, ".png")].startswith("the spooled file's sha256 is ")


def test_an_upload_whose_bytes_changed_in_transit_fails_and_its_file_stays_held(monkeypatch: pytest.MonkeyPatch) -> None:
    async def upload(target, source) -> Uploaded:
        return Uploaded(size_bytes=len(SHOT), sha256="0" * 64)

    with TestClient(_shooting().create_app()) as client:
        _ended(client, _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")]))
        monkeypatch.setattr(_media_spool, "upload", upload)
        pushed = _media(client, {_path(SHOT, ".png"): "image/png"}).json()
        assert _spool(client).held("ctx-1", _path(SHOT, ".png")) is not None
    assert pushed == {"objects": {}, "missing": [], "failed": {_path(SHOT, ".png"): f"the upload's sha256 is {'0' * 64}"}}


def test_a_file_the_spool_cant_write_is_reported_missing_and_the_copy_goes_on(
        store: dict[str, bytes], tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    other = _png(2, 2)
    with TestClient(_shooting().create_app()) as client:
        spool = _spool(client)
        spool._folder = tmp_path / "removed"
        _ended(client, _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")]))
        spool._folder = None
        task_id = _send(client, "again", "m-2", extra_parts=[_file(other, "image/png")])
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
        pushed = _media(client, {_path(SHOT, ".png"): "image/png", _path(other, ".png"): "image/png"}).json()
    assert pushed == {"objects": {_path(other, ".png"): {"size_bytes": len(other),
                                                         "sha256": hashlib.sha256(other).hexdigest()}},
                      "missing": [_path(SHOT, ".png")], "failed": {}}
    paths = {part["source"]["path"] for step in document["steps"] for part in step.get("message") or []
             if isinstance(part, dict) and part.get("type") == "image"}
    assert paths == {_path(SHOT, ".png"), _path(other, ".png")}
    assert f"keeps no file for {_path(SHOT, '.png')}" in caplog.text


def test_a_media_call_takes_only_media_paths() -> None:
    with TestClient(_shooting().create_app()) as client:
        for path in ("../../etc/passwd", "media/abc.png", "https://elsewhere.example.test/x.png",
                     f"media/{'a' * 64}/x.png"):
            refused = _media(client, {path: "image/png"})
            assert refused.status_code == 400, (path, refused.text)


def test_only_an_agent_that_serves_atif_offers_media() -> None:
    def methods(agent: AgentEnvAgent) -> set[str]:
        card = TestClient(agent.create_app()).get("/.well-known/agent-card.json").json()
        [trajectory] = [ext for ext in card["capabilities"]["extensions"] if ext["uri"] == TRAJECTORY_V1.uri]
        return set(trajectory["params"]["methods"])

    assert methods(_shooting()) == {"get", "media"}
    assert methods(_shooting(atif=False)) == {"get"}


def test_a_video_and_a_heic_image_are_spooled_and_pushed_like_any_file(store: dict[str, bytes]) -> None:
    video, heic = b"\x00\x00\x00\x18ftypmp42" + b"v" * 64, b"\x00\x00\x00\x18ftypheic" + b"h" * 64
    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(video, "video/mp4"), _file(heic, "image/heic")])
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
        pushed = _media(client, {_path(video, ".mp4"): "video/mp4", _path(heic, ".heic"): "image/heic"}).json()
    assert errors_in(validate(document)) == []
    [user] = [step for step in document["steps"] if step["source"] == "user"]
    # ATIF has no part for either, so the typed content names each file in a plain note.
    assert f"[video: {_path(video, '.mp4')}, 76 B]" in user["message"]
    assert f"[image: {_path(heic, '.heic')}, 76 B]" in user["message"]
    assert sorted(pushed["objects"]) == sorted([_path(video, ".mp4"), _path(heic, ".heic")])
    assert store[_path(video, ".mp4")] == video and store[_path(heic, ".heic")] == heic


def test_a_context_evicted_or_restored_over_drops_its_media_but_what_another_holds() -> None:
    trajectories = TaskTrajectories(max_contexts=1, native=NATIVE, atif=Shots)

    def play(task_id: str, context_id: str, data: bytes) -> None:
        log = trajectories.register(task_id, context_id)
        trajectories.start(task_id, TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy",
                                              agent_version="1", agent_model=None))
        log.append({"type": "shot", "step": "s", "data": base64.b64encode(data).decode(), "media_type": "image/png"})
        trajectories.seal(task_id, TrajectoryState.COMPLETED, final=(TextPart(text="done"),))

    other = _png(1, 1)
    play("a1", "context-a", SHOT)
    file = trajectories.media.held("context-a", _path(SHOT, ".png"))
    play("b1", "context-b", other)
    assert trajectories.media.held("context-a", _path(SHOT, ".png")) is None and not file.exists()
    play("b2", "context-b", SHOT)
    trajectories.restored("context-b", 0)
    assert trajectories.media.held("context-b", _path(other, ".png")) is None
    trajectories.media.close()


def test_the_spool_goes_with_the_server(store: dict[str, bytes]) -> None:
    with TestClient(_shooting().create_app()) as client:
        _ended(client, _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")]))
        folder = _spool(client).folder
        assert folder.is_dir()
    assert not folder.exists()


def test_the_native_trajectory_is_unchanged_by_media(store: dict[str, bytes]) -> None:
    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")])
        _ended(client, task_id)
        served = client.post("/ext/trajectory", json={"task_id": task_id}).json()["trajectory"]
    assert served == [{"type": "shot", "step": "s0", "data": base64.b64encode(SHOT).decode(), "media_type": "image/png"}]


def _jpeg(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\x00" * 8 + b"\xff\xd9"


def _webp(kind: bytes, body: bytes) -> bytes:
    chunk = kind + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


@pytest.mark.parametrize("data, media_type, size", [
    (_png(640, 480), "image/png", (640, 480)),
    (_jpeg(1280, 720), "image/jpeg", (1280, 720)),
    (b"GIF89a" + struct.pack("<HH", 32, 16) + b"\x00" * 8, "image/gif", (32, 16)),
    (_webp(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", 300, 200) + b"\x00" * 8), "image/webp", (300, 200)),
    (_webp(b"VP8L", b"\x2f" + struct.pack("<I", (99) | (49 << 14)) + b"\x00" * 8), "image/webp", (100, 50)),
    (_webp(b"VP8X", b"\x00" * 4 + (1919).to_bytes(3, "little") + (1079).to_bytes(3, "little")), "image/webp",
     (1920, 1080)),
    (b"\x89PNG not really", "image/png", None),
    (_webp(b"VP8X", b""), "image/webp", None),
    (_webp(b"VP8X", b"\x00" * 4 + b"\x01\x02"), "image/webp", None),
    (b"\xff\xd8\xff\xda", "image/jpeg", None),
    (_png(8, 8), "image/heic", None),
], ids=["png", "jpeg", "gif", "webp-lossy", "webp-lossless", "webp-extended", "bad-png", "truncated-webp",
        "half-a-webp-size", "jpeg-without-frame",
        "not-an-atif-image-type"])
def test_an_images_size_is_read_from_its_header(data: bytes, media_type: str, size: tuple[int, int] | None) -> None:
    assert image_size(data, media_type) == size


def test_dimensions_an_adapter_passes_win_over_the_header() -> None:
    trajectories = TaskTrajectories(native=NATIVE, atif=Shots)
    log = trajectories.register("t1", "ctx")
    trajectories.start("t1", TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy",
                                       agent_version="1", agent_model=None))
    log.append({"type": "shot", "step": "s", "data": base64.b64encode(SHOT).decode(), "media_type": "image/png",
                "info": {"width": 40, "height": 30}})
    trajectories.seal("t1", TrajectoryState.COMPLETED, final=(TextPart(text="done"),))
    document = trajectories.atif_task("t1").document
    media = next(step["extra"]["agentenv"]["media"] for step in document["steps"] if step["source"] == "agent")
    assert (media[_path(SHOT, ".png")]["width"], media[_path(SHOT, ".png")]["height"]) == (40, 30)
    trajectories.media.close()


def test_a_reader_takes_a_part_path_only_as_media_sha256_and_only_with_the_bytes_that_name() -> None:
    path = _path(SHOT, ".png")
    assert media_file(path) == hashlib.sha256(SHOT).hexdigest()
    check_media(path, SHOT)
    for bad in ("../media/x.png", "media/abc.png", "https://example.test/media/x.png", f"/{path}", f"{path}/x"):
        with pytest.raises(AtifError):
            media_file(bad)
    with pytest.raises(AtifError, match="bytes have sha256"):
        check_media(path, SHOT + b"tampered")
