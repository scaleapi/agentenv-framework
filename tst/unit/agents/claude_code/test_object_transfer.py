"""The object forms of skill-config, trajectory and snapshot: agent-env's grants drive the transfer against a fake
object store."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tarfile
from pathlib import Path


from tst.unit.agents.claude_code.support import card, extension_methods, read_grant, send, write_grant
from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge

SKILL_CONFIG_PATH = a2a_server.SKILL_CONFIG_V1.endpoint
TRAJECTORY_PATH = a2a_server.TRAJECTORY_V1.endpoint
SNAPSHOT_PATH = a2a_server.SNAPSHOT_V1.endpoint


def test_card_offers_every_object_form_and_no_s3_form(app):
    skill = extension_methods(app, a2a_server.SKILL_CONFIG_V1.uri)["add"]["request"]
    assert skill["oneOf"] == [{"required": ["skill_md"]}, {"required": ["skill_bundle"]}]
    trajectory = extension_methods(app, a2a_server.TRAJECTORY_V1.uri)["get"]["request"]
    assert {"required": ["task_id", "objects"]} in trajectory["oneOf"]
    assert {"required": ["context_id", "objects"]} in trajectory["oneOf"]
    snapshot = extension_methods(app, a2a_server.SNAPSHOT_V1.uri)
    assert snapshot["load"]["request"] == {"required": ["objects"], "optional": ["target_context_id"]}
    assert set(snapshot) == {"save", "load"}
    requests = json.dumps([
        method.get("request")
        for ext in card(app)["capabilities"]["extensions"]
        for method in ext["params"].get("methods", {}).values()
    ])
    for s3_form in ("skill_s3_url", "trajectory_s3_prefix", "s3_prefix", "presigned_post", "up_to_tool_call_exclusive"):
        assert s3_form not in requests


# ── skills ───────────────────────────────────────────────────────────────────────


def test_a_skill_bundle_is_installed_through_its_read_grants(monkeypatch, tmp_path, app, client, store):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    store.objects.update({"skills/demo/SKILL.md": b"# Demo\n", "skills/demo/scripts/run.sh": b"echo hi\n"})
    bundle = {
        "max_total_bytes": 16,
        "files": [
            {"path": "SKILL.md", "object": read_grant("skills/demo/SKILL.md", 7)},
            {"path": "scripts/run.sh", "object": read_grant("skills/demo/scripts/run.sh", 8)},
        ],
    }

    response = client.post(SKILL_CONFIG_PATH, json={"name": "demo", "description": "d", "skill_bundle": bundle})

    assert response.json() == {"name": "demo"}
    assert (tmp_path / "skills" / "demo" / "scripts" / "run.sh").read_bytes() == b"echo hi\n"
    assert client.get(SKILL_CONFIG_PATH).json()["skills"]["demo"] == {"description": "d"}
    assert {"id": "skill-demo", "name": "demo", "description": "d", "tags": ["skill"]} in card(app)["skills"]


def test_a_skill_bundle_that_fails_leaves_no_skill_behind(monkeypatch, tmp_path, client, store):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    bundle = {"max_total_bytes": 7, "files": [{"path": "SKILL.md", "object": read_grant("missing/SKILL.md", 7)}]}

    response = client.post(SKILL_CONFIG_PATH, json={"name": "gone", "description": "d", "skill_bundle": bundle})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "transfer_rejected"
    assert not (tmp_path / "skills" / "gone").exists()
    assert "gone" not in client.get(SKILL_CONFIG_PATH).json()["skills"]


# ── trajectory ───────────────────────────────────────────────────────────────────


def test_a_task_trajectory_is_uploaded_through_its_grant_and_kept(monkeypatch, client, store):
    trajectory = [{"type": "assistant", "message": "hi"}]

    async def _run_claude(**kwargs):
        for event in trajectory:
            kwargs["event_callback"](event)
        return claude_bridge.ClaudeResult(response="hi", tool_call_count=0, session_id=kwargs["session_id"], trajectory=trajectory)

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    task_id = send(client, "go")["id"]

    result = client.post(
        TRAJECTORY_PATH,
        json={"task_id": task_id, "objects": {"trajectory": write_grant("traj/trajectory-t-1.json", "application/json")}},
    ).json()

    body = store.objects["traj/trajectory-t-1.json"]
    assert body == json.dumps(trajectory, separators=(",", ":")).encode()
    assert result == {"objects": {"trajectory": {"size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}}}
    assert client.post(TRAJECTORY_PATH, json={"task_id": task_id}).json() == {"trajectory": trajectory}


def test_a_context_trajectory_is_uploaded_through_its_grant(monkeypatch, tmp_path, agent, client, store):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    message = {"role": "assistant", "content": [{"type": "text", "text": "working"}]}
    (tmp_path / "s-1.jsonl").write_text(json.dumps({"type": "assistant", "message": message, "sessionId": "s-1"}) + "\n")
    agent.set_session_ref_for_context("ctx-1", "s-1")

    result = client.post(
        TRAJECTORY_PATH,
        json={"context_id": "ctx-1", "objects": {"trajectory": write_grant("traj/ctx-1.json", "application/json")}},
    ).json()

    body = store.objects["traj/ctx-1.json"]
    assert body == json.dumps([{"type": "assistant", "message": message, "session_id": "s-1"}], separators=(",", ":")).encode()
    assert result == {"objects": {"trajectory": {"size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}}}


# ── snapshot ─────────────────────────────────────────────────────────────────────


def _roots(monkeypatch, tmp_path: Path, session_id: str, conversation: bytes) -> Path:
    workspace = tmp_path / "app"
    state = tmp_path / "home" / ".claude"
    session_dir = state / "projects" / "-app"
    session_dir.mkdir(parents=True)
    (session_dir / f"{session_id}.jsonl").write_bytes(conversation)
    (workspace / "report").mkdir(parents=True)
    (workspace / "report" / "answer.txt").write_text("done")
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "CLAUDE_STATE_DIR", state)
    monkeypatch.setattr(a2a_server, "SESSION_JSONL_DIR", session_dir)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", session_dir)
    return workspace


def test_a_snapshot_saves_and_loads_through_grants(monkeypatch, tmp_path, agent, client, store):
    conversation = (json.dumps({"type": "user", "sessionId": "s-1", "message": "hello"}) + "\n").encode()
    workspace = _roots(monkeypatch, tmp_path, "s-1", conversation)
    agent.set_session_ref_for_context("ctx-1", "s-1")

    saved = client.post(
        SNAPSHOT_PATH,
        json={"context_id": "ctx-1", "objects": {"trajectory": write_grant("snap/trajectory"), "workspace": write_grant("snap/workspace")}},
    ).json()

    assert set(saved) == {"context_id", "objects"} and saved["context_id"] == "ctx-1"
    assert store.objects["snap/trajectory"] == conversation
    assert saved["objects"]["workspace"]["size_bytes"] == len(store.objects["snap/workspace"])
    with tarfile.open(fileobj=io.BytesIO(store.objects["snap/workspace"]), mode="r:gz") as archive:
        assert "report/answer.txt" in archive.getnames()

    (workspace / "report" / "answer.txt").unlink()
    loaded = client.put(
        SNAPSHOT_PATH,
        json={
            "objects": {
                "trajectory": read_grant("snap/trajectory", len(conversation)),
                "workspace": read_grant("snap/workspace", len(store.objects["snap/workspace"])),
            },
            "target_context_id": "restored",
        },
    ).json()

    assert loaded == {"context_id": "restored"}
    assert (workspace / "report" / "answer.txt").read_text() == "done"
    new_session = agent.session_ref_for_context("restored")
    restored = (a2a_server.SESSION_JSONL_DIR / f"{new_session}.jsonl").read_text()
    assert json.loads(restored)["sessionId"] == new_session


def test_a_snapshot_load_without_a_workspace_restores_the_conversation_only(monkeypatch, tmp_path, agent, client, store):
    conversation = (json.dumps({"type": "user", "sessionId": "old"}) + "\n").encode()
    _roots(monkeypatch, tmp_path, "s-0", b"")
    store.objects["snap/trajectory"] = conversation

    loaded = client.put(SNAPSHOT_PATH, json={"objects": {"trajectory": read_grant("snap/trajectory", len(conversation))}}).json()

    assert set(loaded) == {"context_id"} and agent.session_ref_for_context(loaded["context_id"]) is not None


def test_a_snapshot_load_refuses_a_bound_context_and_a_running_turn(monkeypatch, tmp_path, agent, client, store):
    conversation = (json.dumps({"type": "user", "sessionId": "old"}) + "\n").encode()
    _roots(monkeypatch, tmp_path, "s-0", b"")
    store.objects["snap/trajectory"] = conversation
    body = {"objects": {"trajectory": read_grant("snap/trajectory", len(conversation))}, "target_context_id": "bound"}
    agent.set_session_ref_for_context("bound", "s-0")

    bound = client.put(SNAPSHOT_PATH, json=body)
    agent._active_contexts.add("other")
    running = client.put(SNAPSHOT_PATH, json={**body, "target_context_id": "fresh"})

    assert (bound.status_code, bound.json()) == (
        409, {"detail": "context_id 'bound' is already bound — retry with a different target_context_id"})
    assert (running.status_code, running.json()) == (
        409, {"detail": "cannot load a workspace while an agent turn is running"})
    assert agent.session_ref_for_context("fresh") is None


def test_a_snapshot_save_and_load_keep_their_400s(client):
    save = client.post(SNAPSHOT_PATH, json={"context_id": "", "objects": {"trajectory": write_grant("snap/trajectory")}})
    load = client.put(
        SNAPSHOT_PATH, json={"objects": {"trajectory": read_grant("snap/trajectory", 1)}, "target_context_id": ""}
    )

    assert (save.status_code, save.json()) == (400, {"detail": "context_id is required"})
    assert (load.status_code, load.json()) == (400, {"detail": "target_context_id must be a non-empty string"})


def test_a_turn_that_arrives_during_a_load_waits_for_it(monkeypatch, agent):
    """The load holds the agent's context lock across its restore, so a turn cannot start
    against a half-restored workspace, and it resumes the session the load bound."""
    started: list[tuple[str, bool]] = []

    async def _run_claude(**kwargs):
        started.append((kwargs["session_id"], kwargs["resume"]))
        return claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    request = a2a_server.TaskRequest(
        task_id="t-1", context_id="restored", parts=(a2a_server.TextPart(text="go"),),
        config=a2a_server.ClaudeCodeConfig(),
    )

    async def scenario():
        async with agent._contexts_lock:
            turn = asyncio.create_task(agent.run(request))
            await asyncio.sleep(0.1)
            assert started == []
            transcript = claude_bridge.session_jsonl_path("restored-session")
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.write_text('{"type": "user", "sessionId": "restored-session"}\n')
            agent.set_session_ref_for_context("restored", "restored-session")
        return await turn

    result = asyncio.run(scenario())

    assert started == [("restored-session", True)] and result.session_ref == "restored-session"


# ── errors ───────────────────────────────────────────────────────────────────────


def test_a_malformed_grant_is_a_validation_400(client):
    response = client.put(SNAPSHOT_PATH, json={"objects": {"trajectory": {"media_type": "x"}}})

    assert response.status_code == 400
    errors = json.loads(response.json()["detail"])
    assert ("objects", "trajectory", "max_bytes") in {tuple(error["loc"]) for error in errors}


def test_a_transfer_error_answers_with_the_typed_body(client, store):
    """A malformed grant is refused by validation; an I/O failure behind a well-formed one is agent-env's typed
    error body, never the grant."""
    response = client.put(SNAPSHOT_PATH, json={"objects": {"trajectory": read_grant("snap/missing", 1)}})

    assert response.status_code == 502
    assert response.json() == {
        "error": {"code": "transfer_rejected", "message": "The object store rejected the transfer.", "retryable": False}
    }
    assert "sig=r" not in response.text


def test_a_snapshot_load_keeps_the_transcripts_of_the_other_conversations(monkeypatch, tmp_path, agent, client, store):
    conversation = (json.dumps({"type": "user", "sessionId": "s-1", "message": "hello"}) + "\n").encode()
    _roots(monkeypatch, tmp_path, "s-1", conversation)
    agent.set_session_ref_for_context("ctx-1", "s-1")
    saved = client.post(SNAPSHOT_PATH, json={"context_id": "ctx-1", "objects": {
        "trajectory": write_grant("snap/trajectory"), "workspace": write_grant("snap/workspace")}})
    assert saved.status_code == 200, saved.text
    other = a2a_server.SESSION_JSONL_DIR / "s-other.jsonl"
    other.write_text(json.dumps({"type": "assistant", "sessionId": "s-other", "message": {"content": "mine"}}) + "\n")
    agent.set_session_ref_for_context("ctx-other", "s-other")

    loaded = client.put(SNAPSHOT_PATH, json={"objects": {
        "trajectory": read_grant("snap/trajectory", len(conversation)),
        "workspace": read_grant("snap/workspace", len(store.objects["snap/workspace"]))}})

    assert loaded.status_code == 200, loaded.text
    assert json.loads(other.read_text())["sessionId"] == "s-other"
    assert client.post(TRAJECTORY_PATH, json={"context_id": "ctx-other"}).status_code == 200


def test_a_snapshot_load_keeps_the_live_transcript_of_a_conversation_the_archive_holds_an_older_copy_of(
        monkeypatch, tmp_path, agent, client, store):
    conversation = (json.dumps({"type": "user", "sessionId": "s-1", "message": "hello"}) + "\n").encode()
    _roots(monkeypatch, tmp_path, "s-1", conversation)
    agent.set_session_ref_for_context("ctx-1", "s-1")
    saved = client.post(SNAPSHOT_PATH, json={"context_id": "ctx-1", "objects": {
        "trajectory": write_grant("snap/trajectory"), "workspace": write_grant("snap/workspace")}})
    assert saved.status_code == 200, saved.text
    live = a2a_server.session_jsonl_path("s-1")
    later = json.dumps({"type": "assistant", "sessionId": "s-1", "message": {"content": "after the save"}}) + "\n"
    live.write_text(live.read_text() + later)

    loaded = client.put(SNAPSHOT_PATH, json={"objects": {
        "trajectory": read_grant("snap/trajectory", len(conversation)),
        "workspace": read_grant("snap/workspace", len(store.objects["snap/workspace"]))}})

    assert loaded.status_code == 200, loaded.text
    assert live.read_text().endswith(later)
