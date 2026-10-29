from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

from agentenv_protocol.transfers import Uploaded

from tst.unit.agents.claude_code.support import extension_methods, read_grant, write_grant
from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge

SNAPSHOT_PATH = a2a_server.SNAPSHOT_V1.endpoint


def _snapshot_roots(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "app"
    state = tmp_path / "home" / ".claude"
    (state / "projects" / "-app" / "memory").mkdir(parents=True)
    (state / "projects" / "-app" / "memory" / "MEMORY.md").write_text(
        "remember this"
    )
    (workspace / "report").mkdir(parents=True)
    (workspace / "report" / "answer.txt").write_text("done")
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "CLAUDE_STATE_DIR", state)
    return workspace, state


def test_agent_card_advertises_unified_save_method(app):
    methods = extension_methods(app, a2a_server.SNAPSHOT_V1.uri)
    assert "save-workspace" not in methods
    assert methods["save"] == {
        "method": "POST",
        "request": {"required": ["context_id", "objects"]},
        "response": {"required": ["context_id", "objects"]},
    }


def test_workspace_archive_includes_hidden_and_nested_files(monkeypatch, tmp_path):
    _snapshot_roots(monkeypatch, tmp_path)
    archive = tmp_path / "workspace.tar.gz"

    file_count, bytes_written = a2a_server._build_workspace_archive(archive)

    assert file_count == 2
    assert bytes_written == archive.stat().st_size
    with tarfile.open(archive, mode="r:gz") as tar:
        names = {member.name for member in tar.getmembers() if member.isfile()}
    assert names == {
        ".claude/projects/-app/memory/MEMORY.md",
        "report/answer.txt",
    }


def test_workspace_archive_tolerates_live_file_races(monkeypatch, tmp_path):
    workspace, _ = _snapshot_roots(monkeypatch, tmp_path)
    archive = tmp_path / "workspace.tar.gz"
    unreadable = workspace / "unreadable.txt"
    unreadable.write_text("secret")
    unstable = workspace / "report" / "answer.txt"
    real_add = tarfile.TarFile.add
    real_copy = tarfile.copyfileobj
    real_open = Path.open
    poisoned = False

    class _FailingReader:
        def __init__(self, source):
            self.source = source
            self.read_once = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.source.close()

        def read(self, size):
            if self.read_once:
                raise OSError("file changed during read")
            self.read_once = True
            return self.source.read(min(size, 1))

    def _add(tar, path, *args, **kwargs):
        if Path(path) == unreadable:
            raise PermissionError("permissions changed")
        return real_add(tar, path, *args, **kwargs)

    def _copy(source, target, length=None, exception=OSError, bufsize=None):
        nonlocal poisoned
        if source.name == str(unstable) and not poisoned:
            poisoned = True
            target.write(source.read(1))
            raise OSError("unexpected end of data")
        return real_copy(
            source,
            target,
            length,
            exception=exception,
            bufsize=bufsize,
        )

    def _open(path, mode="r", *args, **kwargs):
        if path == unreadable and mode == "rb":
            raise PermissionError("permissions changed")
        source = real_open(path, mode, *args, **kwargs)
        if path == unstable and mode == "rb":
            return _FailingReader(source)
        return source

    monkeypatch.setattr(tarfile.TarFile, "add", _add)
    monkeypatch.setattr(tarfile, "copyfileobj", _copy)
    monkeypatch.setattr(Path, "open", _open)

    file_count, _ = a2a_server._build_workspace_archive(archive)

    assert poisoned is True
    assert file_count == 1
    with tarfile.open(archive, mode="r:gz") as tar:
        names = {member.name for member in tar.getmembers() if member.isfile()}
    assert names == {".claude/projects/-app/memory/MEMORY.md"}


def test_save_trims_an_incomplete_tail_takes_no_lock_and_uploads_the_bundle(monkeypatch, tmp_path, agent):
    _, state = _snapshot_roots(monkeypatch, tmp_path)
    session_dir = state / "projects" / "-app"
    session_id = "session-1"
    complete_record = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": session_id,
                "message": {"role": "assistant", "content": "working"},
            }
        )
        + "\n"
    )
    (session_dir / f"{session_id}.jsonl").write_text(
        complete_record + '{"type":"assistant"'
    )
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", session_dir)
    agent.set_session_ref_for_context("ctx-1", session_id)
    uploads: list[dict] = []

    async def _upload(target, source):
        body = source.read_bytes()
        uploads.append({"target": target, "path": source, "body": body})
        return Uploaded(size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

    monkeypatch.setattr(a2a_server, "upload", _upload)
    request = a2a_server.ObjectSnapshotSaveRequest.model_validate(
        {"context_id": "ctx-1", "objects": {"trajectory": write_grant("snap/trajectory"), "workspace": write_grant("snap/workspace")}}
    )

    async def _save_while_a_turn_holds_the_lock():
        # A live save must not wait for the turn it snapshots.
        async with agent._contexts_lock:
            return await asyncio.wait_for(agent.save_snapshot(request), 30)

    result = asyncio.run(_save_while_a_turn_holds_the_lock())

    assert [upload["target"] for upload in uploads] == [request.objects.trajectory, request.objects.workspace]
    assert uploads[0]["body"] == complete_record.encode()
    with tarfile.open(fileobj=io.BytesIO(uploads[1]["body"]), mode="r:gz") as tar:
        assert {member.name for member in tar.getmembers() if member.isfile()} == {
            ".claude/projects/-app/memory/MEMORY.md",
            ".claude/projects/-app/session-1.jsonl",
            "report/answer.txt",
        }
    assert result.model_dump(mode="json", exclude_none=True) == {
        "context_id": "ctx-1",
        "objects": {
            "trajectory": {"size_bytes": len(uploads[0]["body"]), "sha256": hashlib.sha256(uploads[0]["body"]).hexdigest()},
            "workspace": {"size_bytes": len(uploads[1]["body"]), "sha256": hashlib.sha256(uploads[1]["body"]).hexdigest()},
        },
    }
    assert all(not upload["path"].exists() for upload in uploads)


def test_load_restores_workspace_claude_state_and_conversation(monkeypatch, tmp_path, agent, client, store):
    _, source_state = _snapshot_roots(monkeypatch, tmp_path / "source")
    source_session_id = "source-session"
    source_session_dir = source_state / "projects" / "-app"
    conversation = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": source_session_id,
                "message": {"role": "assistant", "content": "remember"},
            }
        )
        + "\n"
    ).encode()
    (source_session_dir / f"{source_session_id}.jsonl").write_bytes(conversation)
    archive = tmp_path / "workspace.tar.gz"
    a2a_server._build_workspace_archive(archive)

    destination_workspace = tmp_path / "destination" / "app"
    destination_state = tmp_path / "destination" / "home" / ".claude"
    destination_session_dir = destination_state / "projects" / "-app"
    destination_workspace.mkdir(parents=True)
    destination_state.mkdir(parents=True)
    (destination_workspace / "stale.txt").write_text("remove me")
    (destination_state / "stale.txt").write_text("remove me")
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", destination_workspace)
    monkeypatch.setattr(a2a_server, "CLAUDE_STATE_DIR", destination_state)
    monkeypatch.setattr(a2a_server, "SESSION_JSONL_DIR", destination_session_dir)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", destination_session_dir)
    store.objects.update({"captures/one/workspace": archive.read_bytes(), "captures/one/trajectory": conversation})

    response = client.put(
        SNAPSHOT_PATH,
        json={
            "objects": {
                "trajectory": read_grant("captures/one/trajectory", len(conversation)),
                "workspace": read_grant("captures/one/workspace", archive.stat().st_size),
            },
            "target_context_id": "restored-context",
        },
    )

    restored_session_id = agent.session_ref_for_context("restored-context")
    restored_jsonl = (
        destination_session_dir / f"{restored_session_id}.jsonl"
    ).read_text()
    assert response.status_code == 200
    assert response.json() == {"context_id": "restored-context"}
    assert [request.url.path for request in store.requests] == [
        "/captures/one/workspace",
        "/captures/one/trajectory",
    ]
    assert (destination_workspace / "report" / "answer.txt").read_text() == "done"
    assert not (destination_workspace / "stale.txt").exists()
    assert not (destination_workspace / ".claude").exists()
    assert (
        destination_state / "projects" / "-app" / "memory" / "MEMORY.md"
    ).read_text() == "remember this"
    assert not (destination_state / "stale.txt").exists()
    assert json.loads(restored_jsonl)["sessionId"] == restored_session_id


def test_load_conversation_only_snapshot_leaves_workspace_untouched(
    monkeypatch, tmp_path, agent, client, store
):
    destination_workspace = tmp_path / "app"
    destination_state = tmp_path / "home" / ".claude"
    destination_session_dir = destination_state / "projects" / "-app"
    destination_workspace.mkdir(parents=True)
    destination_state.mkdir(parents=True)
    (destination_workspace / "keep.txt").write_text("workspace stays")
    (destination_state / "keep.txt").write_text("state stays")
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", destination_workspace)
    monkeypatch.setattr(a2a_server, "CLAUDE_STATE_DIR", destination_state)
    monkeypatch.setattr(a2a_server, "SESSION_JSONL_DIR", destination_session_dir)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", destination_session_dir)

    conversation = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "legacy-session",
                "message": {"role": "assistant", "content": "remember"},
            }
        )
        + "\n"
    ).encode()
    store.objects["captures/legacy/trajectory"] = conversation

    response = client.put(
        SNAPSHOT_PATH,
        json={
            "objects": {"trajectory": read_grant("captures/legacy/trajectory", len(conversation))},
            "target_context_id": "restored-context",
        },
    )

    restored_session_id = agent.session_ref_for_context("restored-context")
    restored_jsonl = (
        destination_session_dir / f"{restored_session_id}.jsonl"
    ).read_text()
    assert response.json() == {"context_id": "restored-context"}
    assert [request.url.path for request in store.requests] == ["/captures/legacy/trajectory"]
    assert (destination_workspace / "keep.txt").read_text() == "workspace stays"
    assert (destination_state / "keep.txt").read_text() == "state stays"
    assert json.loads(restored_jsonl)["sessionId"] == restored_session_id


def test_load_fetch_failure_is_a_typed_502(agent, client, store):
    response = client.put(
        SNAPSHOT_PATH,
        json={"objects": {"trajectory": read_grant("captures/missing/trajectory", 10)}},
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {"code": "transfer_rejected", "message": "The object store rejected the transfer.", "retryable": False}
    }


def test_a_directory_link_is_archived_as_a_link_and_one_the_load_would_refuse_is_left_out(monkeypatch, tmp_path):
    workspace, _ = _snapshot_roots(monkeypatch, tmp_path)
    (workspace / "current").symlink_to("report", target_is_directory=True)
    (workspace / "elsewhere").symlink_to("/etc", target_is_directory=True)
    (workspace / "up").symlink_to("../..", target_is_directory=True)
    archive = tmp_path / "workspace.tar.gz"

    a2a_server._build_workspace_archive(archive)
    with tarfile.open(archive, mode="r:gz") as tar:
        links = {member.name: member.linkname for member in tar.getmembers() if member.issym()}
    for child in list(workspace.iterdir()):
        if child.is_symlink():
            child.unlink()
    a2a_server._restore_workspace_archive(archive)

    assert links == {"current": "report"}
    assert (workspace / "current").is_symlink() and (workspace / "current" / "answer.txt").read_text() == "done"


def test_a_named_pipe_is_left_out_so_the_workspace_still_loads(monkeypatch, tmp_path):
    workspace, _ = _snapshot_roots(monkeypatch, tmp_path)
    os.mkfifo(workspace / "report" / "pipe")
    archive = tmp_path / "workspace.tar.gz"

    file_count, _ = a2a_server._build_workspace_archive(archive)
    (workspace / "report" / "pipe").unlink()
    a2a_server._restore_workspace_archive(archive)

    assert file_count == 2
    assert (workspace / "report" / "answer.txt").read_text() == "done" and not (workspace / "report" / "pipe").exists()


def test_a_saved_conversation_holds_no_key_an_api_error_echoed(monkeypatch, tmp_path):
    _, state = _snapshot_roots(monkeypatch, tmp_path)
    key = "sk-FAKEfake0000fake0000fak"
    ok = json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "working"}})
    failed = json.dumps({"type": "assistant", "isApiErrorMessage": True, "error": "unknown", "message": {
        "role": "assistant", "content": [{"type": "text", "text": f"API Error: 400 Budget exceeded! Key={key}"}]}})
    transcript = state / "projects" / "-app" / "session-1.jsonl"
    transcript.write_text(f"{ok}\n{failed}\n")

    saved = a2a_server._read_complete_session_jsonl(transcript).decode()

    assert key not in saved and "[redacted]" in saved and saved.startswith(f"{ok}\n")
