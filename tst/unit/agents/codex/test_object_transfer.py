"""The object forms of skill-config, trajectory and snapshot: agent-env's grants drive the
transfer against a fake object store."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import transfers
from agentenv_protocol.a2a_agent import (
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
)
from starlette.exceptions import HTTPException

from tst.unit.agents.codex.agent import a2a_server, codex_bridge

ORIGIN = "https://bucket.s3.amazonaws.com"
THREAD_ID = "11111111-1111-1111-1111-111111111111"


def _expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")


def _write(key: str, media_type: str = "application/octet-stream", max_bytes: int = 1 << 20) -> dict:
    return {
        "media_type": media_type,
        "max_bytes": max_bytes,
        "write": {
            "kind": "http-put",
            "url": f"{ORIGIN}/{key}?sig=w",
            "expires_at": _expiry(),
            "headers": {"Content-Type": media_type},
        },
    }


def _read(key: str, size: int | None, max_bytes: int | None = None) -> dict:
    return {
        "media_type": "application/octet-stream",
        "max_bytes": max_bytes or max(size or 0, 1),
        "size_bytes": size,
        "sha256": None,
        "read": {"kind": "http-get", "url": f"{ORIGIN}/{key}?sig=r", "expires_at": _expiry(), "headers": None},
    }


class FakeStore:
    """An object store behind grants: PUT stores bytes, GET serves them."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = request.read()
        key = request.url.path.lstrip("/")
        if request.method == "PUT":
            self.objects[key] = body
            return httpx.Response(200)
        if key not in self.objects:
            return httpx.Response(404, stream=httpx.ByteStream(b"<Error><Code>NoSuchKey</Code></Error>"))
        return httpx.Response(200, stream=httpx.ByteStream(self.objects[key]))


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(transfers, "_RETRY_BACKOFF_SECONDS", 0)
    real_async_client = httpx.AsyncClient

    def async_client(*args, **kwargs):
        # Every grant transfer, the framework's own upload of a completed task's trajectory included.
        kwargs.setdefault("transport", httpx.MockTransport(fake))
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    return fake


def _call(codex, method: str, path: str, body: dict | None = None) -> httpx.Response:
    async def exercise():
        async with codex.client() as client:
            return await client.request(method, path, json=body)

    return asyncio.run(exercise())


def _methods(codex, uri: str) -> dict:
    ext = next(e for e in codex.card["capabilities"]["extensions"] if e["uri"] == uri)
    return ext["params"]["methods"]


def test_card_offers_only_the_object_forms(codex):
    skill = _methods(codex, "urn:agentenv:skill-config/v1")["add"]["request"]
    assert skill == {
        "required": ["name", "description"],
        "oneOf": [{"required": ["skill_md"]}, {"required": ["skill_bundle"]}],
    }
    trajectory = _methods(codex, "urn:agentenv:trajectory/v1")["get"]["request"]
    assert trajectory == {
        "oneOf": [
            {"required": ["task_id"]},
            {"required": ["task_id", "objects"]},
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
            {"required": ["task_id", "after"], "optional": ["limit"]},
            {"required": ["context_id", "after"], "optional": ["limit"]},
        ]
    }
    snapshot = _methods(codex, "urn:agentenv:snapshot/v1")
    assert snapshot["save"]["request"] == {"required": ["context_id", "objects"]}
    assert snapshot["save"]["response"] == {"required": ["context_id", "objects"]}
    assert snapshot["load"]["request"] == {"required": ["objects"], "optional": ["target_context_id"]}
    assert snapshot["load"]["response"] == {"required": ["context_id"]}
    assert "enable-changelog" not in snapshot and "apply-changelog" not in snapshot


# ── skills ───────────────────────────────────────────────────────────────────────


def test_a_skill_bundle_is_installed_through_its_read_grants(codex, monkeypatch, tmp_path, store):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    store.objects.update({"skills/demo/SKILL.md": b"# Demo\n", "skills/demo/scripts/run.sh": b"echo hi\n"})
    bundle = {
        "max_total_bytes": 16,
        "files": [
            {"path": "SKILL.md", "object": _read("skills/demo/SKILL.md", 7)},
            {"path": "scripts/run.sh", "object": _read("skills/demo/scripts/run.sh", 8)},
        ],
    }

    response = _call(codex, "POST", "/ext/skill-config", {"name": "demo", "description": "d", "skill_bundle": bundle})

    assert response.status_code == 200
    assert response.json() == {"name": "demo"}
    assert (tmp_path / "skills" / "demo" / "SKILL.md").read_bytes() == b"# Demo\n"
    assert (tmp_path / "skills" / "demo" / "scripts" / "run.sh").read_bytes() == b"echo hi\n"
    assert {"id": "skill-demo", "name": "demo", "description": "d", "tags": ["skill"]} in codex.card["skills"]
    assert _call(codex, "GET", "/ext/skill-config").json()["skills"]["demo"] == {"description": "d"}


def test_a_skill_bundle_that_fails_leaves_no_skill_behind(codex, monkeypatch, tmp_path, store):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    store.objects["skills/gone/SKILL.md"] = b"# Gone\n"
    bundle = {
        "max_total_bytes": 16,
        "files": [
            {"path": "SKILL.md", "object": _read("skills/gone/SKILL.md", 7)},
            {"path": "missing.txt", "object": _read("skills/gone/missing.txt", 7)},
        ],
    }

    response = _call(codex, "POST", "/ext/skill-config", {"name": "gone", "description": "d", "skill_bundle": bundle})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "transfer_rejected"
    assert ORIGIN not in response.text
    assert not (tmp_path / "skills" / "gone").exists()
    assert "gone" not in _call(codex, "GET", "/ext/skill-config").json()["skills"]


def test_an_inline_skill_is_written_to_the_codex_skills_dir(codex, monkeypatch, tmp_path):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")

    response = _call(codex, "POST", "/ext/skill-config", {"name": "inl", "description": "d", "skill_md": "# Inl\n"})

    assert (response.status_code, response.json()) == (200, {"name": "inl"})
    assert (tmp_path / "skills" / "inl" / "SKILL.md").read_text() == "# Inl\n"
    assert _call(codex, "GET", "/ext/skill-config").json()["skills"] == {
        "codex": {"description": "Runs the OpenAI Codex CLI with full tool access (shell, file edits, web)."},
        "inl": {"description": "d"},
    }


@pytest.mark.parametrize("body, status, detail", [
    ({"name": "empty", "description": "d", "skill_md": "  \n"}, 400, "Skill content must not be empty"),
    ({"name": "a..b", "description": "d", "skill_md": "# x"}, 400,
     "Invalid skill name 'a..b': must not contain path separators or '..'"),
])
def test_a_refused_inline_skill_leaves_no_directory(codex, monkeypatch, tmp_path, body, status, detail):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")

    response = _call(codex, "POST", "/ext/skill-config", body)

    assert (response.status_code, response.json()) == (status, {"detail": detail})
    assert not (tmp_path / "skills" / body["name"]).exists()


def test_a_skill_already_on_disk_or_registered_is_a_conflict(codex, monkeypatch, tmp_path):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    (tmp_path / "skills" / "baked").mkdir(parents=True)
    body = {"description": "d", "skill_md": "# x"}

    assert _call(codex, "POST", "/ext/skill-config", {"name": "baked", **body}).status_code == 409
    assert _call(codex, "POST", "/ext/skill-config", {"name": "fresh", **body}).status_code == 200
    again = _call(codex, "POST", "/ext/skill-config", {"name": "fresh", **body})
    assert (again.status_code, again.json()) == (409, {"detail": "Skill 'fresh' is already registered"})


# ── trajectory ───────────────────────────────────────────────────────────────────


def test_a_task_trajectory_upload_does_not_consume_it(codex, monkeypatch, store):
    events = [{"type": "thread.started", "thread_id": THREAD_ID}, {"type": "turn.completed"}]

    async def fake_run_codex(**kwargs):
        for event in events:
            kwargs["event_callback"](event)
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id=THREAD_ID, trajectory=events)

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            task = await codex.send(client, "go")
            inline = await client.post("/ext/trajectory", json={"task_id": task["id"]})
            granted = await client.post("/ext/trajectory", json={
                "task_id": task["id"],
                "objects": {"trajectory": _write("traj/trajectory.json", "application/json")},
            })
            again = await client.post("/ext/trajectory", json={"task_id": task["id"]})
            return inline.json(), granted.json(), again.json()

    inline, granted, again = asyncio.run(exercise())

    body = store.objects["traj/trajectory.json"]
    assert inline == again == {"trajectory": events}
    assert body == json.dumps(events, separators=(",", ":")).encode()
    assert granted == {"objects": {"trajectory": {"size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}}}


@pytest.mark.parametrize("live", [True, False])
def test_a_context_trajectory_is_uploaded_through_its_grant(codex, monkeypatch, store, live):
    events = [{"type": "thread.started", "thread_id": THREAD_ID}, {"type": "item.completed"}]

    async def exercise():
        gate = asyncio.Event()
        if not live:
            gate.set()
        codex.answer_turns(monkeypatch, events, gate=gate)
        async with codex.client() as client:
            task = await codex.send(client, "go", context_id="ctx-1", blocking=not live)
            async with asyncio.timeout(5):
                while (await client.post("/ext/trajectory", json={"task_id": task["id"], "after": 0})).json()["next"] < 2:
                    await asyncio.sleep(0.01)
            uploaded = await client.post("/ext/trajectory", json={
                "context_id": "ctx-1", "objects": {"trajectory": _write("traj/ctx.json", "application/json")},
            })
            inline = await client.post("/ext/trajectory", json={"context_id": "ctx-1"})
            gate.set()
            return uploaded.json(), inline.json()

    result, inline = asyncio.run(exercise())

    assert store.objects["traj/ctx.json"] == json.dumps(events, separators=(",", ":")).encode()
    assert result == {"objects": {"trajectory": {
        "size_bytes": len(store.objects["traj/ctx.json"]),
        "sha256": hashlib.sha256(store.objects["traj/ctx.json"]).hexdigest(),
    }}}
    assert (inline["trajectory"], inline["is_live"]) == (events, live)


def test_the_s3_trajectory_form_is_refused(codex):
    response = _call(codex, "POST", "/ext/trajectory", {"context_id": "ctx-1", "trajectory_s3_prefix": "s3://b/p"})
    assert response.status_code == 400


# ── snapshot ─────────────────────────────────────────────────────────────────────


def _codex_home(codex, monkeypatch, tmp_path: Path) -> tuple[Path, Path, bytes]:
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "report").mkdir()
    (workspace / "report" / "answer.txt").write_text("done")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    rollout = sessions / f"rollout-test-{THREAD_ID}.jsonl"
    rollout_bytes = (
        json.dumps({"type": "session_meta", "payload": {"id": THREAD_ID}})
        + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "task_started"}})
        + "\n"
    ).encode()
    rollout.write_bytes(rollout_bytes)
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(a2a_server, "find_rollout_path", lambda _: rollout)
    codex.agent.set_session_ref_for_context("ctx-1", THREAD_ID)
    return workspace, sessions, rollout_bytes


def _save_body(context_id: str = "ctx-1", prefix: str = "snap") -> dict:
    return {
        "context_id": context_id,
        "objects": {"trajectory": _write(f"{prefix}/trajectory"), "workspace": _write(f"{prefix}/workspace")},
    }


def _save(codex) -> dict:
    return _call(codex, "POST", "/ext/snapshot", _save_body()).json()


def _load_body(store: FakeStore, *names: str, **extra) -> dict:
    return {
        "objects": {name: _read(f"snap/{name}", len(store.objects[f"snap/{name}"]), 1 << 30) for name in names},
        **extra,
    }


def _restored_rollout(sessions: Path, thread_id: str) -> bytes:
    (path,) = sessions.rglob(f"rollout-*-{thread_id}.jsonl")
    return path.read_bytes()


def test_a_snapshot_saves_and_loads_through_grants(codex, monkeypatch, tmp_path, store):
    workspace, sessions, rollout_bytes = _codex_home(codex, monkeypatch, tmp_path)

    saved = _save(codex)

    assert set(saved) == {"context_id", "objects"} and saved["context_id"] == "ctx-1"
    assert [(r.method, r.url.path) for r in store.requests] == [
        ("PUT", "/snap/workspace"), ("PUT", "/snap/trajectory"),
    ]
    assert store.objects["snap/trajectory"] == rollout_bytes
    for name in ("trajectory", "workspace"):
        body = store.objects[f"snap/{name}"]
        assert saved["objects"][name] == {"size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    with tarfile.open(fileobj=io.BytesIO(store.objects["snap/workspace"]), mode="r:gz") as archive:
        assert "report/answer.txt" in archive.getnames()

    (workspace / "report" / "answer.txt").write_text("mutated")
    (workspace / "stale.txt").write_text("remove me")
    loaded = _call(codex, "PUT", "/ext/snapshot", _load_body(store, "trajectory", "workspace", target_context_id="restored"))

    assert loaded.json() == {"context_id": "restored"}
    assert (workspace / "report" / "answer.txt").read_text() == "done"
    assert not (workspace / "stale.txt").exists()
    new_thread = codex.agent.session_ref_for_context("restored")
    first, rest = _restored_rollout(sessions, new_thread).split(b"\n", 1)
    assert json.loads(first)["payload"]["id"] == new_thread != THREAD_ID
    assert rest == rollout_bytes.split(b"\n", 1)[1]


def test_a_save_without_a_workspace_uploads_an_empty_one_that_loads_as_conversation_only(
    codex, monkeypatch, tmp_path, store
):
    workspace, sessions, rollout_bytes = _codex_home(codex, monkeypatch, tmp_path)
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", tmp_path / "absent")

    saved = _save(codex)

    assert store.objects["snap/workspace"] == b""
    assert saved["objects"]["workspace"]["size_bytes"] == 0

    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    loaded = _call(codex, "PUT", "/ext/snapshot", _load_body(store, "trajectory", "workspace")).json()

    assert (workspace / "report" / "answer.txt").read_text() == "done"
    assert _restored_rollout(sessions, codex.agent.session_ref_for_context(loaded["context_id"]))


def test_a_save_without_a_workspace_grant_uploads_the_trajectory_only(codex, monkeypatch, tmp_path, store):
    _, _, rollout_bytes = _codex_home(codex, monkeypatch, tmp_path)

    saved = _call(codex, "POST", "/ext/snapshot", {
        "context_id": "ctx-1", "objects": {"trajectory": _write("snap/trajectory")},
    }).json()

    assert set(saved["objects"]) == {"trajectory"}
    assert set(store.objects) == {"snap/trajectory"}


def test_a_load_without_a_workspace_object_restores_the_conversation_only(codex, monkeypatch, tmp_path, store):
    workspace, sessions, rollout_bytes = _codex_home(codex, monkeypatch, tmp_path)
    store.objects["snap/trajectory"] = rollout_bytes

    loaded = _call(codex, "PUT", "/ext/snapshot", _load_body(store, "trajectory")).json()

    assert codex.agent.session_ref_for_context(loaded["context_id"])
    assert (workspace / "report" / "answer.txt").read_text() == "done"
    assert [r.method for r in store.requests] == ["GET"]


def test_a_rejected_load_releases_its_locks_and_rollout(codex, monkeypatch, tmp_path, store):
    _, sessions, rollout_bytes = _codex_home(codex, monkeypatch, tmp_path)
    store.objects["snap/trajectory"] = rollout_bytes
    request = _load_body(store, "trajectory", target_context_id="restored")
    request["objects"]["workspace"] = _read("snap/missing", 10, 1 << 30)

    response = _call(codex, "PUT", "/ext/snapshot", request)

    assert response.status_code == 502 and response.json()["error"]["code"] == "transfer_rejected"
    assert codex.agent.session_ref_for_context("restored") is None
    assert "restored" not in a2a_server._context_locks
    assert not a2a_server._workspace_barrier._writer
    assert list(sessions.rglob("rollout-*.jsonl")) == [sessions / f"rollout-test-{THREAD_ID}.jsonl"]


@pytest.mark.parametrize("size", [20, None])
def test_a_transcript_over_the_agent_limit_is_refused_with_the_typed_body(codex, monkeypatch, store, size):
    monkeypatch.setattr(a2a_server, "_MAX_TRANSCRIPT_BYTES", 10)
    store.objects["snap/trajectory"] = b"x" * 20

    response = _call(codex, "PUT", "/ext/snapshot", {"objects": {"trajectory": _read("snap/trajectory", size, 1 << 30)}})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "transfer_too_large"
    assert ORIGIN not in response.text
    assert len(store.requests) == (0 if size is not None else 1)


def test_a_bad_grant_is_refused_before_any_work(codex, store):
    grant = _write("snap/trajectory")
    grant["write"]["kind"] = "http-get"

    response = _call(codex, "POST", "/ext/snapshot", {"context_id": "unknown", "objects": {"trajectory": grant}})

    assert response.status_code == 400
    assert "http-put" in response.text and ORIGIN not in response.text
    assert store.requests == []


@pytest.mark.parametrize("method, body", [
    ("POST", {"context_id": "ctx-1", "s3_prefix": "s3://bucket/snapshots"}),
    ("POST", {**_save_body(), "presigned_post": {"url": ORIGIN, "fields": {}}}),
    ("PUT", {"s3_prefix": "s3://bucket/snapshots"}),
])
def test_the_s3_snapshot_forms_are_refused(codex, store, method, body):
    assert _call(codex, method, "/ext/snapshot", body).status_code == 400
    assert store.requests == []


def test_snapshot_locking(codex, monkeypatch, tmp_path, store):
    """Save and load take the context lock and the workspace barrier exclusively, with
    today's 409s, and a cancelled load releases both."""
    workspace, _, _ = _codex_home(codex, monkeypatch, tmp_path)
    (workspace / "artifact.txt").write_text("original")
    agent = codex.agent
    save = agent.save_snapshot
    load = agent.load_snapshot

    async def exercise():
        saved = await save(ObjectSnapshotSaveRequest.model_validate(_save_body()))
        load_body = _load_body(store, "trajectory", "workspace")

        def load_request(target: str) -> ObjectSnapshotLoadRequest:
            return ObjectSnapshotLoadRequest.model_validate({**load_body, "target_context_id": target})

        source_lock = await a2a_server._get_context_lock("ctx-1")
        await source_lock.acquire()
        try:
            with pytest.raises(HTTPException, match="session is mid-turn; retry after current turn completes"):
                await save(ObjectSnapshotSaveRequest.model_validate(_save_body(prefix="mid-turn")))
        finally:
            source_lock.release()

        busy_lock = await a2a_server._get_context_lock("busy-context")
        await busy_lock.acquire()
        try:
            with pytest.raises(HTTPException, match="target context is active; retry snapshot load later"):
                await load(load_request("busy-context"))
        finally:
            busy_lock.release()

        await a2a_server._workspace_barrier.acquire_shared()
        waiting_save = asyncio.create_task(save(ObjectSnapshotSaveRequest.model_validate(_save_body(prefix="busy"))))
        await asyncio.sleep(0.01)
        assert not waiting_save.done()
        await a2a_server._workspace_barrier.release_shared()
        try:
            waited = await waiting_save
        except BaseException:
            waiting_save.cancel()
            raise
        assert waited["objects"]["workspace"]["size_bytes"] > 0

        monkeypatch.setattr(a2a_server, "_SNAPSHOT_QUIESCE_TIMEOUT_S", 0.01)
        await a2a_server._workspace_barrier.acquire_shared()
        try:
            with pytest.raises(HTTPException, match="another context is mid-turn; retry after it completes"):
                await save(ObjectSnapshotSaveRequest.model_validate(_save_body(prefix="timed-out")))
            with pytest.raises(
                HTTPException, match="an agent turn is active; retry snapshot load after it completes"
            ):
                await load(load_request("quiesce-load"))
        finally:
            await a2a_server._workspace_barrier.release_shared()
        assert "quiesce-load" not in a2a_server._context_locks
        monkeypatch.setattr(a2a_server, "_SNAPSHOT_QUIESCE_TIMEOUT_S", 90)

        await a2a_server._workspace_barrier.acquire_exclusive()
        try:
            cancel_lock = await a2a_server._get_context_lock("cancel-load")
            load_task = asyncio.create_task(load(load_request("cancel-load")))
            for _ in range(1000):
                if cancel_lock.locked():
                    break
                await asyncio.sleep(0.001)
            assert cancel_lock.locked()
            load_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await load_task
        finally:
            await a2a_server._workspace_barrier.release_exclusive()
        assert not cancel_lock.locked()
        assert "cancel-load" not in a2a_server._context_locks

        (workspace / "artifact.txt").write_text("mutated")
        loaded = await load(load_request("restored-context"))
        with pytest.raises(HTTPException, match="context_id 'restored-context' is already bound"):
            await load(load_request("restored-context"))
        return saved, loaded

    saved, loaded = asyncio.run(exercise())
    assert saved["context_id"] == "ctx-1"
    assert loaded == {"context_id": "restored-context"}
    assert (workspace / "artifact.txt").read_text() == "original"


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("gate", ["context lock", "workspace barrier"])
def test_a_snapshot_cancelled_as_its_gate_opens_stays_cancelled(codex, monkeypatch, tmp_path, store, operation, gate):
    """A cancel landing in the loop iteration that frees the context lock or the workspace
    barrier still cancels the save or load, and gives the gate back: asyncio.wait_for returned
    the just-finished acquire instead, and the snapshot went on uploading or restoring."""
    workspace, sessions, _ = _codex_home(codex, monkeypatch, tmp_path)
    (workspace / "artifact.txt").write_text("original")
    monkeypatch.setattr(a2a_server, "_LOCK_TIMEOUT_S", 90)
    agent = codex.agent
    context_id = "ctx-1" if operation == "save" else "cancelled"

    async def exercise():
        await agent.save_snapshot(ObjectSnapshotSaveRequest.model_validate(_save_body()))
        (workspace / "artifact.txt").write_text("mutated")
        rollouts = sorted(sessions.rglob("rollout-*.jsonl"))
        barrier = a2a_server._workspace_barrier
        lock = await a2a_server._get_context_lock(context_id)
        reached = asyncio.Event()
        get_context_lock = a2a_server._get_context_lock

        async def get_context_lock_and_signal(requested: str) -> asyncio.Lock:
            held = await get_context_lock(requested)
            reached.set()
            return held

        monkeypatch.setattr(a2a_server, "_get_context_lock", get_context_lock_and_signal)
        if gate == "context lock":
            await lock.acquire()
        else:
            await barrier.acquire_exclusive()
        if operation == "save":
            call = agent.save_snapshot(ObjectSnapshotSaveRequest.model_validate(_save_body(prefix="cancelled")))
        else:
            body = _load_body(store, "trajectory", "workspace", target_context_id=context_id)
            call = agent.load_snapshot(ObjectSnapshotLoadRequest.model_validate(body))
        task = asyncio.create_task(call)
        await reached.wait()
        if gate == "context lock":
            lock.release()
        else:
            for _ in range(100):
                if barrier._waiting_writers:
                    break
                await asyncio.sleep(0)
            assert barrier._waiting_writers == 1
            await barrier.release_exclusive()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return lock, rollouts

    lock, rollouts = asyncio.run(exercise())
    barrier = a2a_server._workspace_barrier
    assert not lock.locked()
    assert (barrier._readers, barrier._writer, barrier._waiting_writers) == (0, False, 0)
    assert not any(key.startswith("cancelled/") for key in store.objects)
    assert sorted(sessions.rglob("rollout-*.jsonl")) == rollouts
    assert agent.session_ref_for_context("cancelled") is None
    assert "cancelled" not in a2a_server._context_locks
    assert (workspace / "artifact.txt").read_text() == "mutated"


def test_a_snapshot_of_an_unbound_context_is_404(codex, store):
    response = _call(codex, "POST", "/ext/snapshot", _save_body(context_id="nobody"))
    assert (response.status_code, response.json()) == (404, {"detail": "No active thread for context_id 'nobody'"})
    assert store.requests == []


def test_an_empty_target_context_is_refused(codex, monkeypatch, tmp_path, store):
    _, _, rollout_bytes = _codex_home(codex, monkeypatch, tmp_path)
    store.objects["snap/trajectory"] = rollout_bytes

    response = _call(codex, "PUT", "/ext/snapshot", _load_body(store, "trajectory", target_context_id=""))

    assert (response.status_code, response.json()) == (400, {"detail": "target_context_id must be a non-empty string"})
