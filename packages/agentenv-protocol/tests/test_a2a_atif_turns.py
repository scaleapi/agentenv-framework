"""An atif/1 copy's turn index survives what the copy doesn't: a restarted server, a context evicted from memory, and a
snapshot restored into a fresh agent, each of which starts a new document that goes on counting."""
from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient
from test_a2a_atif_serving import NATIVE, Hooks, _atif, _ended, _Hold, _send, _state, _turn

from agentenv_protocol.a2a_agent import (
    SNAPSHOT_STATE_PARAM,
    SNAPSHOT_V1,
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotLoadResponse,
    ObjectSnapshotSaveRequest,
    ObjectSnapshotSaveResponse,
    SnapshotUploadedObjects,
    TaskRequest,
    TaskResult,
    TextPart,
    TrajectoryState,
    Uploaded,
    a2a_agent,
    enable,
    extension,
    framework,
)
from agentenv_protocol.a2a_agent import _turn_index
from agentenv_protocol.a2a_agent._atif_copy import TurnStart
from agentenv_protocol.a2a_agent._trajectory_log import TaskTrajectories
from agentenv_protocol.atif import errors_in, validate

_IDENTITY = AgentIdentity(name="toy-agent", description="test", version="2.0")


def _snapshotting(*, atif: Any = Hooks, save_hold: _Hold | None = None) -> AgentEnvAgent:
    """The toy agent with a snapshot extension of its own, whose save and load the SDK's state rides beside; a save
    waits at ``save_hold``, as one uploading the conversation it captured would."""
    extensions = (enable(TRAJECTORY_V1, live=True, native=NATIVE, **({"atif": atif} if atif else {})),)

    @a2a_agent(identity=_IDENTITY, extensions=extensions)
    class Snapshotting(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return await _turn(request, _Hold())

        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest) -> ObjectSnapshotSaveResponse:
            if save_hold is not None:
                await save_hold.wait()
            return ObjectSnapshotSaveResponse(context_id=request.context_id,
                                              objects=SnapshotUploadedObjects(trajectory=Uploaded(size_bytes=12)))

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest) -> ObjectSnapshotLoadResponse:
            return ObjectSnapshotLoadResponse(context_id=request.target_context_id or "fork-1")

    return Snapshotting()


def _grant(kind: str, name: str, media_type: str) -> dict[str, Any]:
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    access = "write" if kind == "http-put" else "read"
    return {"media_type": media_type, "max_bytes": 100_000,
            access: {"kind": kind, "url": f"https://objects.example.test/{name}", "expires_at": expires}}


def _users(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [step["extra"]["agentenv"] for step in document["steps"] if step["source"] == "user"]


@pytest.fixture
def objects(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """The object store the snapshot's grants name: uploads land here by name, and downloads read from it."""
    stored: dict[str, bytes] = {}

    async def upload(target, body: bytes) -> Uploaded:
        stored[target.write.url.rsplit("/", 1)[-1]] = body
        return Uploaded(size_bytes=len(body))

    async def download(source, destination: Path) -> None:
        destination.write_bytes(stored[source.read.url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(framework, "upload", upload)
    monkeypatch.setattr(framework, "download", download)
    return stored


def test_a_restarted_server_goes_on_counting_in_a_new_document_of_the_same_session() -> None:
    with TestClient(_snapshotting().create_app()) as client:
        for tag in ("t1", "t2"):
            task_id = _send(client, f"work {tag}", f"m-{tag}")
            _ended(client, task_id)
        before = _atif(client, task_id=task_id)["trajectory"]
    # The same container's home, a new process.
    with TestClient(_snapshotting().create_app()) as client:
        task_id = _send(client, "work t3", "m-t3")
        _ended(client, task_id)
        after = _atif(client, task_id=task_id)["trajectory"]
    assert errors_in(validate(after)) == []
    assert [user["turn"] for user in _users(before)] == [0, 1]
    assert [user["turn"] for user in _users(after)] == [2]
    assert after["session_id"] == before["session_id"] == "ctx-1"
    assert after["trajectory_id"] != before["trajectory_id"]


def test_a_context_evicted_from_memory_goes_on_counting() -> None:
    trajectories = TaskTrajectories(max_contexts=1, native=NATIVE, atif=Hooks)

    def play(task_id: str, context_id: str) -> None:
        log = trajectories.register(task_id, context_id)
        trajectories.start(task_id, TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy",
                                              agent_version="1", agent_model=None))
        log.append({"type": "say", "step": f"{task_id}-1", "text": "done"})
        trajectories.seal(task_id, TrajectoryState.COMPLETED, final=(TextPart(text="done"),))

    play("a1", "context-a")
    first = trajectories.atif_task("a1").document
    play("b1", "context-b")
    assert trajectories.atif_task("a1") is None
    play("a2", "context-a")
    second = trajectories.atif_task("a2").document
    assert [user["turn"] for user in _users(second)] == [1]
    assert second["trajectory_id"] != first["trajectory_id"]


def test_a_snapshot_carries_the_turn_count_to_the_fresh_agent_it_is_restored_into(
        objects: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with TestClient(_snapshotting().create_app()) as client:
        for tag in ("t1", "t2"):
            _ended(client, _send(client, f"work {tag}", f"m-{tag}"))
        saved = client.post("/ext/snapshot", json={"context_id": "ctx-1", "objects": {
            "trajectory": _grant("http-put", "trajectory", "application/octet-stream"),
            "state": _grant("http-put", "state", "application/json")}})
    assert saved.status_code == 200, saved.text
    assert saved.json()["objects"]["state"]["size_bytes"] == len(objects["state"])
    assert json.loads(objects["state"]) == {"turns": 2}
    # Another container: none of the first one's counts.
    monkeypatch.setattr(_turn_index, "TURNS_DIR", tmp_path / "fresh-agent")
    with TestClient(_snapshotting().create_app()) as client:
        loaded = client.put("/ext/snapshot", json={"target_context_id": "ctx-restored", "objects": {
            "trajectory": _grant("http-get", "trajectory", "application/octet-stream"),
            "state": _grant("http-get", "state", "application/json")}})
        assert loaded.status_code == 200, loaded.text
        task_id = _send(client, "work t3", "m-t3", context_id="ctx-restored")
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
    assert errors_in(validate(document)) == []
    assert [(user["turn"], user["context_id"]) for user in _users(document)] == [(2, "ctx-restored")]
    assert document["session_id"] == "ctx-restored"


def test_a_restore_into_a_context_the_agent_holds_starts_a_new_document(objects: dict[str, bytes]) -> None:
    with TestClient(_snapshotting().create_app()) as client:
        _ended(client, _send(client, "work t1", "m-t1"))
        client.post("/ext/snapshot", json={"context_id": "ctx-1", "objects": {
            "trajectory": _grant("http-put", "trajectory", "application/octet-stream"),
            "state": _grant("http-put", "state", "application/json")}})
        task_id = _send(client, "work t2", "m-t2")
        _ended(client, task_id)
        before = _atif(client, task_id=task_id)["trajectory"]
        client.put("/ext/snapshot", json={"target_context_id": "ctx-1", "objects": {
            "trajectory": _grant("http-get", "trajectory", "application/octet-stream"),
            "state": _grant("http-get", "state", "application/json")}})
        task_id = _send(client, "work t3", "m-t3")
        _ended(client, task_id)
        after = _atif(client, task_id=task_id)["trajectory"]
    # The conversation is the snapshot's again, taken after one turn: the count goes on from there.
    assert [user["turn"] for user in _users(before)] == [0, 1]
    assert [user["turn"] for user in _users(after)] == [1]
    assert after["trajectory_id"] != before["trajectory_id"] and after["session_id"] == "ctx-1"


def test_a_snapshot_state_that_isnt_the_sdks_is_refused(objects: dict[str, bytes]) -> None:
    objects["trajectory"], objects["state"] = b"conversation", b'{"turns": -1}'
    with TestClient(_snapshotting().create_app()) as client:
        refused = client.put("/ext/snapshot", json={"target_context_id": "ctx-restored", "objects": {
            "trajectory": _grant("http-get", "trajectory", "application/octet-stream"),
            "state": _grant("http-get", "state", "application/json")}})
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "snapshot_state_invalid", refused.text


def test_an_agent_that_serves_no_atif_writes_no_state(objects: dict[str, bytes]) -> None:
    with TestClient(_snapshotting(atif=None).create_app()) as client:
        saved = client.post("/ext/snapshot", json={"context_id": "ctx-1", "objects": {
            "trajectory": _grant("http-put", "trajectory", "application/octet-stream"),
            "state": _grant("http-put", "state", "application/json")}})
    assert saved.status_code == 200, saved.text
    assert "state" not in saved.json()["objects"] and "state" not in objects


def test_the_card_names_the_state_only_an_sdk_serving_atif_keeps() -> None:
    def snapshot_params(agent: AgentEnvAgent) -> dict[str, Any]:
        card = TestClient(agent.create_app()).get("/.well-known/agent-card.json").json()
        [snapshot] = [ext for ext in card["capabilities"]["extensions"] if ext["uri"] == SNAPSHOT_V1.uri]
        return snapshot["params"]

    assert snapshot_params(_snapshotting())[SNAPSHOT_STATE_PARAM] == ["turns"]
    assert SNAPSHOT_STATE_PARAM not in snapshot_params(_snapshotting(atif=None))


def test_a_task_waits_for_its_contexts_save_so_the_saved_count_is_its_captures(objects: dict[str, bytes]) -> None:
    paused = _Hold()
    with TestClient(_snapshotting(save_hold=paused).create_app()) as client:
        for tag in ("t1", "t2"):
            _ended(client, _send(client, f"work {tag}", f"m-{tag}"))
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            saving = pool.submit(client.post, "/ext/snapshot", json={"context_id": "ctx-1", "objects": {
                "trajectory": _grant("http-put", "trajectory", "application/octet-stream"),
                "state": _grant("http-put", "state", "application/json")}})
            paused.reached()
            task_id = _send(client, "work t3", "m-t3")
            time.sleep(0.3)
            waiting = _state(client, task_id)
            paused.resume.set()
            saved = saving.result(timeout=10)
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
    assert waiting not in ("completed", "failed", "canceled")
    assert saved.status_code == 200, saved.text
    assert json.loads(objects["state"]) == {"turns": 2}
    assert [user["turn"] for user in _users(document)] == [0, 1, 2]


def test_two_servers_on_one_home_share_its_counts(tmp_path: Path) -> None:
    first, second = _turn_index.TurnIndex(tmp_path), _turn_index.TurnIndex(tmp_path)
    assert [first.next("ctx"), second.next("ctx"), first.next("ctx")] == [0, 1, 2]


def test_a_count_that_cant_be_read_starts_again_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    turns = _turn_index.TurnIndex(tmp_path)
    turns.set("ctx", 3)
    next(tmp_path.glob("*.json")).write_text("not a count")
    with caplog.at_level(logging.WARNING):
        assert turns.next("ctx") == 0
    assert turns.count("ctx") == 1 and "can't be read" in caplog.text


def test_a_snapshot_state_grant_names_json() -> None:
    with TestClient(_snapshotting().create_app()) as client:
        refused = client.post("/ext/snapshot", json={"context_id": "ctx-1", "objects": {
            "trajectory": _grant("http-put", "trajectory", "application/octet-stream"),
            "state": _grant("http-put", "state", "application/octet-stream")}})
    assert refused.status_code == 400 and "state media_type" in refused.text


def test_a_count_that_cant_be_written_is_kept_in_memory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    turns = _turn_index.TurnIndex(blocked)
    with caplog.at_level(logging.WARNING):
        assert [turns.next("ctx"), turns.next("ctx"), turns.next("other")] == [0, 1, 0]
    assert sum("kept in memory only" in record.message for record in caplog.records) == 1
