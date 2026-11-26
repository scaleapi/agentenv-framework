"""snapshot/v1: a context's conversation, its SDK store whole with the env times its turns were given, and the
workspace, saved and loaded into another context under the conversation's own id; an archive that doesn't hold one is
refused."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
from agentenv_protocol.a2a_agent import Uploaded

from tst.unit.agents.antigravity.agent import a2a_server, antigravity_bridge
from tst.unit.agents.antigravity.conftest import Harness

_GRANT = {"expires_at": "2099-01-01T00:00:00Z", "url": "https://objects.example/o"}
_WRITE = {"media_type": "application/octet-stream", "max_bytes": 1 << 20, "write": {"kind": "http-put", **_GRANT}}
_READ = {"media_type": "application/octet-stream", "max_bytes": 1 << 20, "size_bytes": 2,
         "read": {"kind": "http-get", **_GRANT}}


@pytest.fixture
def objects(monkeypatch) -> dict[str, bytes]:
    """The objects a snapshot uploads, by the name of the file it uploaded, and loads back by the name it downloads
    to."""
    stored: dict[str, bytes] = {}

    async def upload(_target, source: Path):
        stored[source.name] = source.read_bytes()
        return Uploaded(size_bytes=len(stored[source.name]))

    async def download(_source, destination: Path):
        destination.write_bytes(stored[destination.name])

    monkeypatch.setattr(a2a_server, "upload", upload)
    monkeypatch.setattr(a2a_server, "download", download)
    return stored


def _members(archive: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return sorted(tar.getnames())


@pytest.mark.asyncio
async def test_a_conversation_its_own_records_and_its_workspace_load_into_another_agents_context(ag, objects, folders):
    store = antigravity_bridge.SAVE_DIR
    store.mkdir(parents=True)
    (store / "conv-1.db").write_bytes(b"sqlite")
    (store / "conv-1.db-wal").write_bytes(b"wal")
    (store / "conv-2.db").write_bytes(b"another")
    antigravity_bridge.ENV_TIMES.put("2026-11-25T10:00:00-08:00", "2031-03-03T23:06:07-06:00")
    antigravity_bridge.TURN_IMAGES.put("3f2a", ["c0ffee.png"])
    antigravity_bridge.file_records("conv-1", {"2026-11-25T10:00:00-08:00"}, {"3f2a"})
    antigravity_bridge.ENV_TIMES.put("2026-11-25T11:00:00-08:00", "2031-04-01T00:00:00Z")
    antigravity_bridge.TURN_IMAGES.put("9b1d", ["0beef.png"])
    antigravity_bridge.file_records("conv-2", {"2026-11-25T11:00:00-08:00"}, {"9b1d"})
    antigravity_bridge.IMAGES_DIR.mkdir()
    for name in ("c0ffee.png", "0beef.png", "unused.png"):
        (antigravity_bridge.IMAGES_DIR / name).write_bytes(b"\x89PNG")
    (folders / "report.md").write_text("draft")
    ag.agent.set_session_ref_for_context("ctx-source", "conv-1")

    async with ag.client() as client:
        saved = await client.post("/ext/snapshot", json={"context_id": "ctx-source", "objects": {
            "trajectory": _WRITE, "workspace": _WRITE}})
        assert saved.status_code == 200, saved.text
        assert _members(objects["conversation.tgz"]) == ["env-times.json", "images/c0ffee.png", "manifest.json",
                                                         "store/conv-1.db", "store/conv-1.db-wal", "turn-images.json"]
        shutil.rmtree(store)
        store.mkdir()
        (folders / "report.md").write_text("changed since")
        (folders / "stray.txt").write_text("stray")

    other = Harness(a2a_server.AntigravityChangelogAgent())
    async with other.client() as client:
        loaded = await client.put("/ext/snapshot", json={"target_context_id": "ctx-target", "objects": {
            "trajectory": _READ, "workspace": _READ}})

    assert loaded.status_code == 200, loaded.text
    assert loaded.json() == {"context_id": "ctx-target"}
    assert other.agent.session_ref_for_context("ctx-target") == "conv-1"
    assert {path.name: path.read_bytes() for path in store.glob("conv-*")} == {"conv-1.db": b"sqlite",
                                                                              "conv-1.db-wal": b"wal"}
    assert antigravity_bridge.ENV_TIMES.all() == {"2026-11-25T10:00:00-08:00": "2031-03-03T23:06:07-06:00"}
    assert antigravity_bridge.TURN_IMAGES.all() == {"3f2a": ["c0ffee.png"]}
    assert [path.name for path in antigravity_bridge.IMAGES_DIR.iterdir()] == ["c0ffee.png"]
    assert antigravity_bridge.CONVERSATION_RECORDS.all() == {
        "conv-1": {"env_times": ["2026-11-25T10:00:00-08:00"], "turn_images": ["3f2a"]}}
    assert sorted(path.name for path in folders.iterdir()) == ["report.md"]
    assert (folders / "report.md").read_text() == "draft"


@pytest.mark.asyncio
async def test_a_context_loaded_with_its_own_conversation_goes_back_to_it(ag, objects):
    store = antigravity_bridge.SAVE_DIR
    store.mkdir(parents=True)
    (store / "conv-1.db").write_bytes(b"saved")
    ag.agent.set_session_ref_for_context("ctx", "conv-1")
    async with ag.client() as client:
        await client.post("/ext/snapshot", json={"context_id": "ctx", "objects": {"trajectory": _WRITE}})
        (store / "conv-1.db").write_bytes(b"later")
        (store / "conv-1.db-shm").write_bytes(b"stale")
        loaded = await client.put("/ext/snapshot", json={"target_context_id": "ctx", "objects": {"trajectory": _READ}})

    assert loaded.status_code == 200, loaded.text
    assert {path.name: path.read_bytes() for path in store.glob("conv-1.db*")} == {"conv-1.db": b"saved"}
    assert ag.agent.session_ref_for_context("ctx") == "conv-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [{"target_context_id": "ctx-other"}, {}])
async def test_a_conversation_another_context_of_the_agent_holds_isnt_loaded_beside_it(ag, objects, target):
    store = antigravity_bridge.SAVE_DIR
    store.mkdir(parents=True)
    (store / "conv-1.db").write_bytes(b"saved")
    ag.agent.set_session_ref_for_context("ctx", "conv-1")
    async with ag.client() as client:
        await client.post("/ext/snapshot", json={"context_id": "ctx", "objects": {"trajectory": _WRITE}})
        (store / "conv-1.db").write_bytes(b"later")
        loaded = await client.put("/ext/snapshot", json={**target, "objects": {"trajectory": _READ}})

    assert loaded.status_code == 409, loaded.text
    assert "load it into another agent" in loaded.json()["detail"]
    assert (store / "conv-1.db").read_bytes() == b"later"
    assert ag.agent.contexts_bound_to("conv-1") == ["ctx"]


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("archive, detail", [
    (_archive({"store/conv-1.db": b"db"}), "snapshot has no conversation manifest"),
    (_archive({"manifest.json": json.dumps({"conversation_id": "../conv-1"}).encode()}),
     "snapshot names no valid conversation id"),
    (_archive({"manifest.json": json.dumps({"conversation_id": "conv-1"}).encode()}),
     "snapshot has no store for its conversation"),
    (_archive({"manifest.json": json.dumps({"conversation_id": "conv-1"}).encode(), "../escape.db": b"x"}),
     "snapshot's conversation isn't an archive this agent reads"),
    (b"not a tarball", "snapshot's conversation isn't an archive this agent reads"),
], ids=["no-manifest", "id-outside-the-store", "no-store", "member-outside", "not-an-archive"])
async def test_an_archive_without_a_conversation_is_refused(ag, objects, archive, detail):
    objects["conversation.tgz"] = archive
    async with ag.client() as client:
        loaded = await client.put("/ext/snapshot", json={"target_context_id": "ctx", "objects": {
            "trajectory": _READ}})

    assert (loaded.status_code, loaded.json()) == (400, {"detail": detail})
    assert ag.agent.session_ref_for_context("ctx") is None
